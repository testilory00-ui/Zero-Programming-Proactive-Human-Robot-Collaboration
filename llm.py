import cv2
import os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import json
import time
import re
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient


class LLM_planner:
    def __init__(self, model_name="qwen3_4B_INT8", assembly_file='learned_memory.json',
                 load_local_model=True):
        # Always load assembly memory (needed by both local and API backends)
        with open(assembly_file, 'r', encoding='utf-8') as f:
            self.memory = json.load(f)
        self.assembly_str = json.dumps(self.memory, indent=2)

        if load_local_model:
            print("\nLoading model and compiling for GPU (this may take a moment)...")
            start_setup = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
            self.model = OVModelForCausalLM.from_pretrained(model_name, device="GPU")
            print(f"Setup Time (Load & Compile): {time.time() - start_setup:.2f} seconds")
        else:
            self.tokenizer = None
            self.model = None
            print("LLM_planner: local model skipped (API-only mode).")

        api_key = os.environ.get("GEMINI_API_KEY")
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the script.")

        if not hf_token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set. Please set it before running the script.")

        self.client = genai.Client(api_key=api_key)
        self.hf_client = InferenceClient(token=hf_token)

        self.system_prompt = "You are an assembly task planner. Output strict JSON only, no explanations."

    @staticmethod
    def _first_json_object(text: str) -> str:
        # Return only the first complete JSON object in `text`. Some backends
        # (HF Llama4, Gemini under hint-injection) occasionally append a second
        # object or trailing prose after the JSON, which makes json.loads raise
        # "Extra data". raw_decode stops at the end of the first value.
        if not isinstance(text, str):
            return text
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        if start == -1:
            return candidate.strip()
        try:
            _, end = json.JSONDecoder().raw_decode(candidate, start)
            return candidate[start:end]
        except json.JSONDecodeError:
            return candidate[start:].strip()

    @staticmethod
    def _split_action_delta(semantic_action: str) -> tuple[str, str]:
        """Split '<action> | recent: <delta>' into (action_line, recent_section).

        The delta is surfaced on its own line with an explicit hint so smaller
        models (Llama4 via HF) don't treat the pipe-suffixed tokens as noise.
        Returns ("", "") if semantic_action is empty.
        """
        if not semantic_action:
            return "", ""
        if " | recent:" not in semantic_action:
            return semantic_action, ""
        action_part, _, recent_part = semantic_action.partition(" | recent:")
        recent_section = f'\nRECENT CHANGE: {recent_part.strip()}\n'
        return action_part.strip(), recent_section

    def infer_LLM(self, scene_data, augmented_hint: str | None = None):
        last_step = len(self.memory)
        steps_simple = "\n".join(
            f"Step {s['step number']}: {s['step description']} [objects_required: {', '.join(s['objects_required'])}]"
            for s in self.memory
        )

        semantic_action = scene_data.get("semantic_action", "unknown")
        step_completion_context = scene_data.get("step_completion_context", "")
        context_section = f"\n{step_completion_context}\n" if step_completion_context else ""
        action_line, recent_section = self._split_action_delta(semantic_action)
        hint_section = f"[SCENE CORRECTION — MUST FOLLOW]\n{augmented_hint}\n\n" if augmented_hint else ""

        if augmented_hint:
            guidance = (
                "A [SCENE CORRECTION] is present above — follow it strictly. "
                "Select only from the compatible steps listed. Do NOT repeat the previous prediction."
            )
        else:
            guidance = "Match the observed action and its objects directly to the step descriptions. Use the STEP TRACKER hint only as a secondary check when the action is ambiguous."

        user_prompt = f"""{hint_section}ASSEMBLY STEPS:
{steps_simple}
{context_section}
OBSERVED ACTION: "{action_line}"
{recent_section}
Select the step whose description best matches the observed action by meaning.
Next step = current + 1 (after step {last_step}, next is step 1).
{guidance}
Return ONLY this JSON:
{{
  "stage of assembly": "<current step description>",
  "next operation": "<next step description>",
  "objects required": [<objects list for NEXT step>]
}}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,  # Qwen3: suppress <think>…</think> block (saves time)
        )

        outputs = self.model.generate(**inputs, max_new_tokens=160)  # JSON is ~50-70 tokens; 160 is safe
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return self._first_json_object(response)

    def infer_LLM_GEMMA(self, scene_data):
        last_step = len(self.memory)
        steps_simple = "\n".join(
            f"Step {s['step number']}: {s['step description']} [objects_required: {', '.join(s['objects_required'])}]"
            for s in self.memory
        )

        semantic_action = scene_data.get("semantic_action", "unknown")
        step_completion_context = scene_data.get("step_completion_context", "")
        context_section = f"\n{step_completion_context}\n" if step_completion_context else ""
        action_line, recent_section = self._split_action_delta(semantic_action)

        prompt = f"""ASSEMBLY STEPS:
{steps_simple}
{context_section}
OBSERVED ACTION: "{action_line}"
{recent_section}
Select the step whose description best matches the observed action by meaning.
Next step = current + 1 (after step {last_step}, next is step 1).
When it's ambiguous, use the PRIOR hint above (if provided) to break ties.
Return ONLY this JSON:
{{
  "stage of assembly": "<current step description>",
  "next operation": "<next step description>",
  "objects required": [<objects list for NEXT step>]
}}"""

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=self.system_prompt,
        )

        # gemini-3.1-flash-lite-preview
        # gemini-2.5-flash-lite
        response = self.client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt,
            config=config,
        )

        return response.text

    def infer_LLM_HF(self, scene_data, augmented_hint: str | None = None):
        """LLM inference via Hugging Face Inference API (serverless)."""
        last_step = len(self.memory)
        steps_simple = "\n".join(
            f"Step {s['step number']}: {s['step description']} [objects_required: {', '.join(s['objects_required'])}]"
            for s in self.memory
        )

        semantic_action = scene_data.get("semantic_action", "unknown")
        step_completion_context = scene_data.get("step_completion_context", "")
        context_section = f"\n{step_completion_context}\n" if step_completion_context else ""
        action_line, recent_section = self._split_action_delta(semantic_action)
        hint_section = f"[SCENE CORRECTION — MUST FOLLOW]\n{augmented_hint}\n\n" if augmented_hint else ""

        if augmented_hint:
            ctx_note = (
                " ALSO use the STEP COMPLETION CONTEXT (if present) to confirm your choice "
                "does not contradict already-completed steps."
            ) if step_completion_context else ""
            guidance = (
                "A [SCENE CORRECTION — MUST FOLLOW] block is present above. "
                "The OBSERVED ACTION below may be WRONG or MISLEADING — treat it as secondary evidence only. "
                "Pick 'next operation' EXCLUSIVELY from the compatible steps listed in the correction. "
                "Set 'stage of assembly' to the step immediately before your chosen 'next operation'. "
                f"Do NOT repeat the previous prediction.{ctx_note}"
            )
        else:
            guidance = (
                "The PRIOR above is derived from real-time tracking and is reliable. "
                "When the action is ambiguous, follow the PRIOR — do not default to an earlier step. "
                "If the PRIOR marks a step as done but the observed action strongly matches it, prefer the next step instead."
            )

        user_prompt = f"""{hint_section}ASSEMBLY STEPS:
{steps_simple}
{context_section}
OBSERVED ACTION: "{action_line}"
{recent_section}
Select the step whose description best matches the observed action by meaning.
Next step = current + 1 (after step {last_step}, next is step 1).
{guidance}
If a RECENT CHANGE line is present, it indicates an object that just entered — usually names the CURRENT step.
Return ONLY this JSON:
{{
  "stage of assembly": "<current step description>",
  "next operation": "<next step description>",
  "objects required": [<objects list for NEXT step>]
}}"""

        # Qwen/Qwen2.5-7B-Instruct
        # Qwen/Qwen3-4B-Instruct-2507
        # meta-llama/Llama-4-Scout-17B-16E-Instruct
        response = self.hf_client.chat_completion(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=160,
            response_format={"type": "json_object"},
        )

        return self._first_json_object(response.choices[0].message.content)

    def infer_VLM(self, frames, step_completion_context="", model='gemini-2.5-flash-lite',
                  augmented_hint: str | None = None,
                  frame_detections: list[list[str]] | None = None):
        _frame_labels = [
            "earliest (before action)",
            "mid-action early phase",
            "mid-action late phase",
            "latest (end of observation window)",
        ]
        contents = []
        for i, frame in enumerate(frames):
            _, buffer = cv2.imencode('.jpg', frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))
            label = _frame_labels[i] if i < len(_frame_labels) else f"frame {i+1}"
            if frame_detections and i < len(frame_detections) and frame_detections[i]:
                det_str = ", ".join(frame_detections[i])
                contents.append(f"[Frame {i+1} — {label}] YOLO detected: {det_str}")
            else:
                contents.append(
                    f"[Frame {i+1} — {label}] YOLO detected: nothing "
                    f"(objects may be occluded, out of frame, or below confidence threshold)")

        config = types.GenerateContentConfig(response_mime_type="application/json")

        context_section = f"\nSTEP COMPLETION CONTEXT:\n{step_completion_context}\n" if step_completion_context else ""

        steps_simple = "\n".join(
            f"Step {s['step number']}: {s['step description']} [objects_required: {', '.join(s['objects_required'])}]"
            for s in self.memory
        )
        last_step = len(self.memory)

        hint_section = f"\n[SCENE CORRECTION — MUST FOLLOW]\n{augmented_hint}\n" if augmented_hint else ""

        step1_desc = self.memory[0]["step description"] if self.memory else "Step 1"
        step1_objs = self.memory[0].get("objects_required", []) if self.memory else []

        prompt = f"""You are a vision module for an industrial assembly task.
{hint_section}
ASSEMBLY STEPS (use this as authoritative reference):
{steps_simple}
{context_section}
You receive 4 frames in chronological order. Each frame is immediately followed by a text annotation listing the objects YOLO detected in that frame. Bounding boxes are also drawn on the frames as visual anchors.
YOLO detections are reliable when present, but may be incomplete: objects can be missed due to occlusion, hand obstruction, or low confidence. If detections are absent in some frames, rely on visual evidence from the other frames and your own visual reasoning to fill the gaps.

IMPORTANT DEFINITIONS:
- "stage_of_assembly" = the assembly step the OPERATOR IS ACTIVELY PERFORMING in these frames (hands visibly picking up, inserting, or fastening a component). It is NOT the inferred completion state of the assembly.
  - If the operator's hands are NOT performing an assembly gesture in the frames, stage_of_assembly MUST be "idle".
  - Do NOT set stage_of_assembly to a step just because its objects appear to be present on the table — objects sitting in storage look the same as objects not yet installed.
- "next_operation" = the step the ROBOT SHOULD PREPARE NEXT.
  - {"If a [SCENE CORRECTION] block is present, pick next_operation exclusively from the compatible steps listed there. Do NOT repeat the previous prediction." if augmented_hint else f"If stage_of_assembly is 'idle' or 'no step detected': use the STEP COMPLETION CONTEXT (if provided) to find the first incomplete step, or default to Step 1 ('{step1_desc}') when no context is available."}
  - If stage_of_assembly matches a step: next_operation = the immediately following step (after step {last_step}, wrap to step 1).

Given the 4 frames:
1. Look at the operator's hands: are they actively handling a component? Describe what they are doing.
2. Set stage_of_assembly: match to a step description only if hands are visibly active on that component; otherwise "idle".
3. Set next_operation following the rules above.
4. Copy objects_required from the step table for next_operation.

Return ONLY this JSON:
{{
  "current_action": "<what the operator's hands are doing, or 'no action'>",
  "stage_of_assembly": "<matched step description, idle, or no step detected>",
  "next_operation": "<next step description from the table, or none>",
  "objects_required": [<objects_required list for next_operation>]
}}"""
        contents.append(prompt)

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response.text)
        return self._first_json_object(cleaned)

    def infer_VLM_learn(self, frames, reference_frame=None):
        """
        Describe what assembly action occurred given 4 frames (before/mid/mid/after).
        reference_frame: YOLO-annotated image used to anchor object names visually.
        Returns {description, objects_required} or None on error.
        """
        contents = []

        # The reference image comes first so the VLM builds a visual name→appearance
        # map before it ever sees the action frames.
        if reference_frame is not None:
            _, buffer = cv2.imencode('.jpg', reference_frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))
            contents.append(
                "REFERENCE IMAGE (above): captured during the initial inventory scan. "
                "YOLO bounding boxes and class labels are drawn on every visible assembly object. "
                "Use these labels as the authoritative object names for the rest of this analysis."
            )

        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))

        ref_instruction = (
            "The first image is the YOLO reference — use its labels to identify objects by name. "
            "The following 4 images are the action sequence.\n"
            if reference_frame is not None
            else ""
        )

        prompt = f"""You are observing a manual assembly task. {ref_instruction}You receive exactly 4 frames
            in chronological order:
              Frame 1: before the action — component not yet mounted, operator reaching.
              Frame 2: operator actively mounting the component (early phase).
              Frame 3: operator actively mounting the component (late phase).
              Frame 4: after the action — component mounted, operator's hand released.

            Using all four frames, describe what assembly action was performed in one concise sentence.
            Identify which objects the operator is using in this specific step, using the exact label
            names shown in the reference image.

            Return ONLY a JSON object in this exact format:
            {{
            "description": "<one sentence describing the action performed>",
            "objects_required": ["object_name_1", "object_name_2"]
            }}

            Rules:
            - Focus on what changed between frame 1 and frame 4 (which component was placed and where).
            - Use ONLY label names visible in the reference image; only use a different name if the object is clearly absent from that reference.
            - An object does not need to be visible in ALL frames to be included — it may be hidden under the operator's hand or partially assembled.
            - CRITICAL: If screws, bolts, or other fasteners are present or being tightened in ANY frame, you MUST include them in objects_required. Fasteners are small and easy to overlook — examine every frame carefully for screws being picked up, inserted, or tightened with a screwdriver. A screwdriver without screws in the list is almost always wrong.
            - If nothing meaningful happened, return description as "no action detected" and an empty list.
            - Do not include speculation or objects you cannot clearly see."""

        contents.append(prompt)
        config = types.GenerateContentConfig(response_mime_type="application/json")

        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"  [VLM Learn Error] {e}")
            return None

    def infer_VLM_learn_inverse(self, frames, reference_frame=None,
                                detected_classes=None):
        """
        Describe what disassembly action occurred given 4 frames (before/mid/mid/after removal).
        reference_frame: YOLO-annotated post-removal frame; detected_classes: authoritative vocabulary.
        Returns {description, objects_required, removed_from} or None on error.
        """
        contents = []

        if reference_frame is not None:
            _, buffer = cv2.imencode('.jpg', reference_frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))
            class_list = ", ".join(detected_classes) if detected_classes else "see bounding boxes"
            contents.append(
                "REFERENCE IMAGE (above): the scene AFTER the removal action. "
                "YOLO bounding boxes are drawn on every detected object. "
                f"Detected object classes: [{class_list}]. "
                "Use these class names exactly in your answer."
            )

        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))

        ref_instruction = (
            "The first image is a YOLO reference taken AFTER the removal. "
            "The 4 images that follow are the chronological action sequence.\n"
            if reference_frame is not None
            else ""
        )

        # Build a class-name hint for the prompt
        if detected_classes:
            class_hint = (
                f"Known object class names: [{', '.join(detected_classes)}]. "
                "You MUST pick from this list for objects_required.\n"
            )
        else:
            class_hint = ""

        prompt = f"""You are watching an operator disassemble a mechanical assembly.
            {ref_instruction}{class_hint}
            The 4 action frames are in chronological order:
              Frame 1: before — assembly is intact, operator is reaching toward it.
              Frame 2: mid-action — operator actively removing a component.
              Frame 3: mid-action — operator completing the removal.
              Frame 4: after — the removed component is now on the table.

            Study all 4 frames. In frames 2 and 3, identify the component the
            operator's hands are gripping or pulling away from the assembly.
            Use frame 1 vs frame 4 to confirm what changed.

            Return ONLY this JSON:
            {{
              "description": "<what was removed, how and from where. One sentence>",
              "objects_required": ["<removed_component>"],
              "removed_from": "<parent component or sub-assembly, or 'assembly' if unclear>"
            }}

            Rules:
            - in "description", "from where" means the part of the assembly where was the removed component
                (e.g "from the bottom", "inside the hole", ...)
            - objects_required: the removed component. Add a tool (e.g. screwdriver)
              ONLY if it is actively used against the component in frame 2 or 3 —
              not just lying on the table.
            - Use the exact class names from the list above.
            - If no removal action is visible, return description "no action detected",
              objects_required [], removed_from ""."""

        contents.append(prompt)
        config = types.GenerateContentConfig(response_mime_type="application/json")

        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"  [VLM Learn Inverse Error] {e}")
            return None

    def infer_VLM_cleanup(self, raw_observations, frame_buffer, object_inventory=None,
                          previous_result=None, operator_feedback=None):
        """
        Deduplicate and order raw assembly observations into clean memory.json steps.
        previous_result + operator_feedback trigger a revision pass.
        Returns a list matching memory.json schema, or None on parse failure.
        """
        n_obs = len(raw_observations)

        frames_by_step = {entry["step_index"]: entry["frames"] for entry in frame_buffer}

        max_images = 12  # cap images to limit token usage; evenly spaced
        if n_obs <= max_images:
            steps_with_image = set(range(n_obs))
        else:
            indices = [round(i * (n_obs - 1) / (max_images - 1)) for i in range(max_images)]
            steps_with_image = set(indices)

        contents = []
        for i, obs in enumerate(raw_observations):
            step_idx = obs.get("raw_step_index", i + 1)
            if i in steps_with_image and step_idx in frames_by_step:
                for action_frame in frames_by_step[step_idx]:
                    _, buf = cv2.imencode('.jpg', action_frame)
                    contents.append(types.Part.from_bytes(
                        data=buf.tobytes(), mime_type='image/jpeg'
                    ))
            objs_str = ", ".join(obs.get("objects_required", []))
            label = f'[Observation {i + 1}]: {obs.get("description", "")} | Objects: {objs_str}'
            contents.append(label)

        if object_inventory:
            inventory_str = ", ".join(object_inventory)
            inventory_hint = f"""
            COMPLETE OBJECT INVENTORY (scanned at session start — these are ALL objects in this assembly):
            {inventory_str}
            When writing objects_required for each step, use ONLY names from this list unless an
            object is clearly visible in the images but absent from the list.
            """
        else:
            inventory_hint = ""

        prompt = f"""You are an expert manufacturing engineer analyzing a recorded assembly procedure.

            INPUT FORMAT:
            You receive a sequence of raw observations, each consisting of:
            - A text description of what the vision system detected.
            - Up to 2 images captured mid-action (operator actively mounting the component).
            Observations are already in chronological order.
            {inventory_hint}
            
            YOUR TASK:
            Produce a clean, minimal, ordered list of distinct assembly steps.

            STEP 1 — FILTER. Discard any observation that is clearly a false detection:
            - Description says "no action detected" or is vague with no specific component.
            - Images show no assembly gesture (hands idle, no object being manipulated).
            - Both text and images are ambiguous with nothing clearly placed or moved.
            Do NOT discard an observation just because the images are blurry — rely on the text if images are unclear.

            STEP 2 — DEDUPLICATE. Consecutive observations describing the same physical action
            (same component placed in the same location) must be merged into one step.
            Use the images to confirm: if two observations show the same object in the same
            position, they are duplicates. Keep the clearer description of the two.

            STEP 3 — NORMALIZE object names: lowercase, underscores instead of spaces.
            Use names from the inventory when they match what is shown; only use a different
            name if you are certain the object is absent from the inventory.

            STEP 4 — WRITE the final steps. Each step must describe one distinct physical
            action (placing, inserting, fastening one component). The description should be
            one concise sentence: what the operator did and which part was involved.

            STEP 5 — COVERAGE CHECK. Every object in the inventory is a physical part
            that must be mounted during the assembly. After writing the steps, verify that
            each inventory object appears in at least one step's objects_required.
            If an object is missing:
            - Look through the observations and images for any step where that object
              was visibly handled or placed — add it to objects_required of that step.
            - If no existing step is a reasonable match, add a new step at the most
              logical position in the sequence (based on what you see in the images).
            - Do NOT silently omit inventory objects. Every one must end up in some step.

            CRITICAL RULES:
            - A screwdriver in objects_required ALWAYS requires the corresponding fastener
              (screw, bolt) in the same step. Never list a screwdriver alone.
            - Do NOT reorder steps unless the chronological order is clearly wrong based
              on the images (e.g., a component appears already placed in an earlier frame).

            Return ONLY a JSON array in this exact format:
            [
            {{
                "step number": <integer starting from 1>,
                "step description": "<one sentence: what was done and which component>",
                "objects_required": ["object_A", "object_B"]
            }}

            Show briefly your reasoning
            ]"""

        if previous_result is not None and operator_feedback:
            prev_json = json.dumps(previous_result, indent=2)
            prompt += f"""

            IMPORTANT — REVISION REQUEST:
            A previous cleanup produced the following result, which the operator rejected:
            {prev_json}

            The operator provided this correction:
            "{operator_feedback}"

            Apply the operator's correction to produce an improved result. Keep everything
            else from the previous result that was not mentioned in the correction."""

        contents.append(prompt)

        config = types.GenerateContentConfig(response_mime_type="application/json")

        try:
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=contents,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[Cleanup VLM] Failed: {e}")
            return None

    def infer_VLM_cleanup_inverse(self, raw_observations, frame_buffer,
                                   object_inventory=None, previous_result=None,
                                   operator_feedback=None):
        """
        Reconstruct assembly steps from disassembly observations (inverse learning).
        Reorders observations using attachment topology to infer assembly order.
        Returns a list matching memory.json schema, or None on parse failure.
        """
        n_obs = len(raw_observations)
        frames_by_step = {entry["step_index"]: entry["frames"] for entry in frame_buffer}

        max_images = 12  # cap images to limit token usage; evenly spaced
        if n_obs <= max_images:
            steps_with_image = set(range(n_obs))
        else:
            indices = [round(i * (n_obs - 1) / (max_images - 1)) for i in range(max_images)]
            steps_with_image = set(indices)

        contents = []
        for i, obs in enumerate(raw_observations):
            step_idx = obs.get("raw_step_index", i + 1)
            if i in steps_with_image and step_idx in frames_by_step:
                for action_frame in frames_by_step[step_idx]:
                    _, buf = cv2.imencode('.jpg', action_frame)
                    contents.append(types.Part.from_bytes(
                        data=buf.tobytes(), mime_type='image/jpeg'
                    ))
            objs_str = ", ".join(obs.get("objects_required", []))
            removed_from = obs.get("removed_from", "unknown")
            label = (f'[Observation {i + 1}]: {obs.get("description", "")} '
                     f'| Objects: {objs_str} '
                     f'| Removed from: {removed_from}')
            contents.append(label)

        if object_inventory:
            inventory_str = ", ".join(object_inventory)
            inventory_hint = f"""
            COMPLETE OBJECT INVENTORY (all objects observed during disassembly):
            {inventory_str}
            Every one of these objects must appear in at least one assembly step.
            When writing objects_required, use ONLY names from this list unless an
            object is clearly visible in the images but absent from the list.
            """
        else:
            inventory_hint = ""

        prompt = f"""You are an expert manufacturing engineer. You are given a sequence of
            DISASSEMBLY observations recorded while an operator took apart a product.
            Your task is to reconstruct the correct ASSEMBLY procedure.

            INPUT FORMAT:
            Each observation describes a removal action with:
            - A text description of what was removed and how.
            - Up to 2 images captured mid-action (operator actively removing the component).
            - "Removed from" indicating what the component was attached to.

            Observations are in chronological disassembly order — but disassembly was
            performed casually, NOT in reverse-assembly order. You MUST reorder for assembly.
            {inventory_hint}
            YOUR TASK:
            Produce a clean, ordered list of ASSEMBLY steps.

            STEP 1 — FILTER. Discard any observation that is clearly a false detection:
            - Description says "no action detected" or is vague with no specific component.
            - Images show no removal gesture (hands idle, no object being manipulated).
            Do NOT discard an observation just because the images are blurry — rely on the text.

            STEP 2 — DEDUPLICATE. Observations describing the same removal
            (same component from same location) must be merged into one entry.
            Keep the clearer description.

            STEP 3 — REORDER FOR ASSEMBLY. Use the "Removed from" relationships as a
            dependency graph to determine the correct assembly order.
            Assembly ordering constraints:
              a) BASE/BODY components first — the main structure everything mounts onto.
              b) INTERNAL components before their enclosures
              c) COVERS/HOUSINGS after the internals they protect.
              d) FASTENERS (screws, bolts) immediately after the component they secure:
                 if step N mounts a cover or other similar components, step N should include the screws too.
              e) PERIPHERAL/EXTERNAL components last,
                 unless they must be installed before a cover blocks access.
              f) When two components have no dependency, preserve the reverse of their
                 disassembly order as a tiebreaker.

            STEP 4 — NORMALIZE object names: lowercase, underscores instead of spaces.
            Use names from the inventory when they match what is shown; only use a different
            name if you are certain the object is absent from the inventory.

            STEP 5 — COVERAGE CHECK. Every object in the inventory is a physical part
            that must be mounted during the assembly. After writing the steps, verify that
            each inventory object appears in at least one step's objects_required.
            If an object is missing:
            - Look through the observations and images for any step where that object
              was visibly handled — add it to objects_required of that step.
            - If no existing step is a reasonable match, add a new step at the most
              logical position in the sequence.
            - Do NOT silently omit inventory objects.

            CRITICAL RULES:
            - A screwdriver in objects_required ALWAYS requires the corresponding fastener
              (screw, bolt) in the same step. Never list a screwdriver alone.
            - if a tool is needed for an operation, put it in the same step.
            - The output must be a valid ASSEMBLY procedure — someone following these steps
              from step 1 to the last step must end up with the fully assembled product.
            - "Removed from" relationships encode attachment topology: a component removed
              from X means X must be assembled BEFORE that component.

            Return ONLY a JSON array in this exact format:
            [
            {{
                "step number": <integer starting from 1>,
                "step description": "<one and concise sentence assembly instruction>",
                "objects_required": ["object_A", "object_B", ...]
            }}
            ]
            
            """

        if previous_result is not None and operator_feedback:
            prev_json = json.dumps(previous_result, indent=2)
            prompt += f"""

            IMPORTANT — REVISION REQUEST:
            A previous cleanup produced the following result, which the operator rejected:
            {prev_json}

            The operator provided this correction:
            "{operator_feedback}"

            Apply the operator's correction to produce an improved result. Keep everything
            else from the previous result that was not mentioned in the correction."""

        contents.append(prompt)

        config = types.GenerateContentConfig(response_mime_type="application/json")

        try:
            response = self.client.models.generate_content(
                model='gemini-3.5-flash',
                contents=contents,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[Cleanup Inverse VLM] Failed: {e}")
            return None

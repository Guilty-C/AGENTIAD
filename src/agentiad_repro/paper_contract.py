"""
Single Source of Truth for AgentIAD Paper Contract (I/O).
Enforces strict XML tagging, schema validation, and loss masking rules.
This module is dependency-light (standard library only where possible) to allow
easy import in inference, training, and verification scripts.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logger
logger = logging.getLogger(__name__)

class PaperContract:
    """
    Defines the exact I/O contract for AgentIAD as specified in the paper.
    """
    
    # -------------------------------------------------------------------------
    # 1. Prompt Templates & Special Tokens
    # -------------------------------------------------------------------------
    
    SYSTEM_PROMPT = (
        "You are an industrial anomaly detection agent. You must analyze the provided image/data "
        "and determine if an anomaly is present. You have access to tools. "
        "You must output your final answer strictly in the following XML format:\n"
        "<answer>\n"
        "{\n"
        '  "anomaly_present": boolean,\n'
        '  "top_anomaly": "string",\n'
        '  "visual_descriptions": ["string", ...]\n'
        "}\n"
        "</answer>\n"
        "For normal cases, set anomaly_present=false, top_anomaly=\"none\", and visual_descriptions=[].\n"
        "Tool calls must be wrapped in <tool_call> tags:\n"
        "<tool_call>\n"
        '{"name": "...", "arguments": {...}}\n'
        "</tool_call>"
    )

    # XML Tags
    TAG_ANSWER_START = "<answer>"
    TAG_ANSWER_END = "</answer>"
    TAG_TOOL_CALL_START = "<tool_call>"
    TAG_TOOL_CALL_END = "</tool_call>"

    # -------------------------------------------------------------------------
    # 2. Parsing Logic
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_answer_xml(text: str) -> Optional[str]:
        """
        Extracts the content between <answer> and </answer> tags.
        Returns None if tags are missing or malformed.
        """
        pattern = f"{PaperContract.TAG_ANSWER_START}(.*?){PaperContract.TAG_ANSWER_END}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_tool_calls_xml(text: str) -> List[Dict[str, Any]]:
        """
        Extracts and parses all content between <tool_call> and </tool_call> tags.
        Returns a list of parsed tool call dictionaries.
        """
        pattern = f"{PaperContract.TAG_TOOL_CALL_START}(.*?){PaperContract.TAG_TOOL_CALL_END}"
        matches = re.findall(pattern, text, re.DOTALL)
        tool_calls = []
        for m in matches:
            try:
                tc = json.loads(m.strip())
                if isinstance(tc, dict):
                    tool_calls.append(tc)
            except Exception:
                pass
        return tool_calls

    @staticmethod
    def parse_model_output(text: str) -> Dict[str, Any]:
        """
        Parses the full model output into a structured dictionary.
        Prioritizes the <answer> block, but also extracts tool calls.
        """
        result = {
            "valid_syntax": False,
            "data": {},
            "tool_calls": [],
            "raw_xml": None,
            "error": None
        }

        # Extract tool calls first (can appear before answer)
        result["tool_calls"] = PaperContract.extract_tool_calls_xml(text)

        # Extract Answer
        raw_xml = PaperContract.extract_answer_xml(text)
        if not raw_xml:
            result["error"] = "Missing <answer> tags"
            # If we have tool calls but no answer, it might be a tool turn.
            return result

        try:
            data = json.loads(raw_xml)
            result["valid_syntax"] = True
            result["data"] = data
            result["raw_xml"] = raw_xml
        except json.JSONDecodeError as e:
            result["error"] = f"Invalid JSON inside <answer>: {str(e)}"
            result["raw_xml"] = raw_xml
            
        return result

    # -------------------------------------------------------------------------
    # 3. Validation Logic
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates the parsed data against the strict paper schema.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        # Required keys
        required_keys = {"anomaly_present", "top_anomaly", "visual_descriptions"}
        missing = required_keys - data.keys()
        if missing:
            errors.append(f"Missing required keys: {missing}")

        # Forbidden legacy keys
        forbidden_keys = {"description", "confidence"}
        found_forbidden = forbidden_keys.intersection(data.keys())
        if found_forbidden:
            errors.append(f"Forbidden legacy keys found: {found_forbidden}")

        # Type checks
        if "anomaly_present" in data and not isinstance(data["anomaly_present"], bool):
            errors.append("Field 'anomaly_present' must be a boolean")
        
        if "top_anomaly" in data and not isinstance(data["top_anomaly"], str):
            errors.append("Field 'top_anomaly' must be a string")
            
        if "visual_descriptions" in data and not isinstance(data["visual_descriptions"], list):
            errors.append("Field 'visual_descriptions' must be a list")
        
        # Normal consistency check
        if data.get("anomaly_present") is False:
            if data.get("top_anomaly") != "none":
                errors.append("If anomaly_present is False, top_anomaly must be 'none'")
            if data.get("visual_descriptions") != []:
                errors.append("If anomaly_present is False, visual_descriptions must be []")

        return len(errors) == 0, errors

    @staticmethod
    def validate_tool_call(tool_call: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a single tool call dictionary against paper contract.
        """
        errors = []
        if "name" not in tool_call or "arguments" not in tool_call:
            errors.append("Tool call must have 'name' and 'arguments' keys")
            return False, errors
            
        name = tool_call["name"]
        args = tool_call["arguments"]
        
        if name == "crop_image_normalized":
            if "bbox_2d" not in args:
                errors.append("crop_image_normalized requires 'bbox_2d'")
            elif not isinstance(args["bbox_2d"], list) or len(args["bbox_2d"]) != 4:
                errors.append("bbox_2d must be a list of 4 floats")
            else:
                for v in args["bbox_2d"]:
                    if not isinstance(v, (int, float)) or v < 0.0 or v > 1.0:
                        errors.append(f"bbox_2d value {v} out of range [0.0, 1.0]")
                if len(errors) == 0:
                     if args["bbox_2d"][2] < args["bbox_2d"][0]:
                         errors.append("bbox_2d x2 < x1")
                     if args["bbox_2d"][3] < args["bbox_2d"][1]:
                         errors.append("bbox_2d y2 < y1")
            
            if "target_image" not in args:
                errors.append("crop_image_normalized requires 'target_image'")
            elif not isinstance(args["target_image"], int) or args["target_image"] < 1:
                errors.append("target_image must be an int >= 1")
                
        elif name == "query_image":
            if args != {}:
                errors.append("query_image arguments must be empty {}")
        else:
            errors.append(f"Unknown tool name: {name}")
            
        return len(errors) == 0, errors

    # -------------------------------------------------------------------------
    # 4. SFT Loss Masking Utilities
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_loss_mask_strategy() -> str:
        return "mask_user_prompts_train_cot_and_answer"

    # -------------------------------------------------------------------------
    # 5. Tool Definitions (for reference)
    # -------------------------------------------------------------------------
    
    ALLOWED_TOOLS = {
        "crop_image_normalized": {
            "params": ["bbox_2d", "target_image"]
        },
        "query_image": {
            "params": []
        }
    }


def _self_test():
    """
    Simple self-test to verify contract logic.
    """
    print("Running PaperContract self-test...")
    
    # Test Case 1: Valid Output
    valid_output = """
    <think>Some reasoning...</think>
    <answer>
    {
        "anomaly_present": true,
        "top_anomaly": "crack",
        "visual_descriptions": ["long crack on surface"]
    }
    </answer>
    """
    parsed = PaperContract.parse_model_output(valid_output)
    assert parsed["valid_syntax"] is True, "Failed to parse valid output"
    is_valid, errors = PaperContract.validate_schema(parsed["data"])
    assert is_valid, f"Failed schema validation: {errors}"
    print("  [PASS] Valid output parsing")

    # Test Case 2: Normal Consistency
    bad_normal = """
    <answer>
    {
        "anomaly_present": false,
        "top_anomaly": "crack",
        "visual_descriptions": []
    }
    </answer>
    """
    parsed = PaperContract.parse_model_output(bad_normal)
    is_valid, errors = PaperContract.validate_schema(parsed["data"])
    assert not is_valid, "Should fail normal consistency check"
    assert "top_anomaly must be 'none'" in errors[0], "Wrong error for normal consistency"
    print("  [PASS] Normal consistency check")

    # Test Case 3: Tool Call Validation
    valid_tool = """
    <tool_call>
    {"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.1,0.1,0.5,0.5], "target_image": 1}}
    </tool_call>
    """
    parsed = PaperContract.parse_model_output(valid_tool)
    assert len(parsed["tool_calls"]) == 1
    is_valid, errors = PaperContract.validate_tool_call(parsed["tool_calls"][0])
    assert is_valid, f"Failed valid tool check: {errors}"
    
    invalid_tool = """
    <tool_call>
    {"name": "query_image", "arguments": {"class_name": "foo"}}
    </tool_call>
    """
    parsed = PaperContract.parse_model_output(invalid_tool)
    is_valid, errors = PaperContract.validate_tool_call(parsed["tool_calls"][0])
    assert not is_valid, "Should fail invalid tool args"
    assert "empty" in errors[0], "Wrong error for query_image args"
    print("  [PASS] Tool validation")

    print("All self-tests passed!")

if __name__ == "__main__":
    _self_test()

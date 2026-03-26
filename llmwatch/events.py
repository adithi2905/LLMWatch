from __future__ import annotations
from dataclasses import dataclass,field, asdict
from typing import Optional
import json
import uuid
import time

@dataclass
class RawRequestEvent:
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def success(cls,
                provider,
                model,
                start_time,
                end_time,
                ttft,
                prompt_tokens,
                completion_tokens,
                prompt,
                response) -> RawRequestEvent:
        
        return cls(
            request_id=str(uuid.uuid4()),
            provider=provider,
            model=model,
            start_time=start_time,
            end_time=end_time,
            ttft=ttft,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt=prompt,
            response=response)
    
    @classmethod
    def error(cls,
            provider,
            model,
            start_time,
            end_time,
            prompt,
            error_message) -> RawRequestEvent:
        return cls(
            request_id=str(uuid.uuid4()),
            provider=provider,
            model=model,
            start_time=start_time,
            end_time=end_time,
            ttft=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            prompt=prompt,
            error_message=error_message
        )
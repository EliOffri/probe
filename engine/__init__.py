import asyncio
import litellm
from litellm import acompletion


async def acompletion_with_retry(max_retries: int = 6, base_delay: float = 1.0, **kwargs):
    """
    Wrapper around litellm.acompletion that retries on 429 rate-limit errors
    with exponential backoff.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await acompletion(**kwargs)
        except litellm.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            import re
            # Honour the retry-after hint from the error message if present
            match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE)
            wait = float(match.group(1)) if match else delay
            # Always wait at least the hinted amount, plus a small buffer
            wait = max(wait, delay) + 0.5
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30.0)

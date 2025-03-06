#!/usr/bin/env python3

import os
import sys
import logging
import random
import json
import re
import time
import datetime
import requests
import concurrent.futures

from typing import List, Dict, Tuple
from collections import Counter

import openai
import anthropic  # Kept for minimal changes – not used anymore in David
from anthropic import Anthropic  # Also kept for minimal changes – no longer used in David
from dotenv import load_dotenv

import fal_client



# ----------------------------------------------------------------
# Setup and Configuration
# ----------------------------------------------------------------

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

load_dotenv()

required_env_vars = [
    "OPENAI_API_KEY",
    "FLUX_API_KEY",
    "ANTHROPIC_API_KEY",
    "FACEBOOK_ACCESS_TOKEN",
    "INSTAGRAM_ACCOUNT_ID"
]
missing_vars = [v for v in required_env_vars if not os.getenv(v)]
if missing_vars:
    logging.warning(f"Missing environment variables: {missing_vars}")

openai.api_key = os.getenv('OPENAI_API_KEY')
flux_api_key = os.getenv('FLUX_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

instagram_access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')

FLUX_API_URL = 'https://api.bfl.ml/v1/flux-pro-1.1-ultra'


# ----------------------------------------------------------------
# Agent Base Class
# ----------------------------------------------------------------

class Agent:
    """
    Base agent class with name and model fields.
    Subclasses handle specialized tasks (analysis, image gen, captioning, etc.).
    """
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model


# ----------------------------------------------------------------
# Tricia: Trend Analyst (Autonomously Uses Date Context)
# ----------------------------------------------------------------

class Tricia(Agent):
    """
    Tricia is responsible for analyzing top-performing posts,
    summarizing them, generating 10 new image prompts, optionally
    weaving in a 'theme', picking a suitable content type,
    AND dynamically asking the model if today's date correlates with
    a major holiday or celebration to inspire the post.
    """

    def generate_image_prompts(
        self,
        image_descriptions: List[str],
        caption_categories: List[str],
        theme: str = "",
        location: str = ""
    ) -> Tuple[List[str], str]:
        logging.info(f"[{self.name}] Generating image prompts based on top-performing posts...")

        # Check for date-based/holiday context with the model:
        autonomous_date_context = self.get_autonomous_date_context()

        # Limit descriptions to first 5
        limited_descriptions = image_descriptions[:5]
        logging.debug(f"[{self.name}] Limited Descriptions: {limited_descriptions}")

        # Summarize them
        summarized_descriptions = self.summarize_descriptions(limited_descriptions)
        logging.debug(f"[{self.name}] Summarized Descriptions: {summarized_descriptions}")

        # Turn them into bullet-points
        descriptions_text = '\n'.join(f"- {desc.strip()}" for desc in summarized_descriptions if desc.strip())
        logging.debug(f"[{self.name}] descriptions_text:\n{descriptions_text}")

        # Convert caption categories to a comma-separated string
        categories_text = ', '.join(set(caption_categories))
        logging.debug(f"[{self.name}] categories_text:\n{categories_text}")

        if not descriptions_text.strip():
            logging.warning(f"[{self.name}] No valid image descriptions provided. Using default inspiration.")
            descriptions_text = "- A serene indoor garden with natural light."

        if not categories_text.strip():
            logging.warning(f"[{self.name}] No caption categories provided. Proceeding without them.")
            categories_text = "Lifestyle Descriptions"

        # We'll inject the holiday/celebration context if the model returned anything
        date_context_note = ""
        if autonomous_date_context:
            date_context_note = (
                "\n\n"
                "Also note the following date-based context derived by analyzing today's date:\n"
                f"\"{autonomous_date_context}\"\n"
            )

        # Define the prompt ALWAYS (avoid UnboundLocalError)
        prompt = f"""
Your job is to plan today’s post on Daily Biophilia, which consists of:
- exactly 10, numbered image prompts – which will be used to generate images using DALLE3, Flux, or Recraft, all proprietary AI image models for an Instagram carousel – simulating a tour of one unconventional, biophilic 2 bedroom apartment
- a corresponding caption to complement the carousel that an audience living in Los Angeles, San Francisco, New York, and San Diego with interests in DIY ideas and interior design would  enjoy

The post should be planned in a way that:
- maximizes engagement as measured by ((accounts engaged over accounts reached) + saves) * shares on Instagram 

You will use the following brief descriptions of Daily Biophilia's recent, top-performing Instagram posts as inspiration for the exactly 10 new image prompts for a biophilic design:

{descriptions_text}

Also, consider the following caption categories that have performed well:

{categories_text}

Each prompt (numbered 1 - 10) should follow this format:

"[#]. The [shot_type] shot of the stunningly awe-inspiring & tearfully joyous [room_type] with [elements_str] & accented with [plant_types] of an [color_scheme] biophilic [style] apartment in [neighborhood] [city] designed by [architects] to moderately activate the SNS"

Where:
- [#] is the number of the prompt, going from 1 to 10 and ordered in a way as we expect the viewer to view them (we are generating a 10 slide Instagram carousel with these images)
- [shot_type] is a specific type of camera shot ('mid-angle detail', 'elevated wide angle', 'close-up detail', '8mm fisheye lens', 'dutch angle', 'wide angle', 'low angle' -- select from these)
- [room_type] is a simple, standard type of room (e.g., 'bedroom', 'living room', 'dining room', 'kitchen', 'bathroom', 'foyer', 'front door', 'backyard porch' etc)
- [elements_str] includes unconventional features (e.g. 'backlit panel of brown agate', 'sculptural bathtub carved from a single piece of black basalt', 'indoor cacti installation', 'color-tinted onyx & marble', 'live-edge wood furniture & rustic modern fusion' -- iterate on MORE of these. be unconventional)
- [plant_types] are specific and unconventional plants available in that region that might best complement the space
- [style] is an unconventional architectural style, consistent across all prompts (e.g. 'art deco', 'memphis', 'brutalist'  -- iterate on MORE of these. be unconventional)
- [neighborhood] is a specific, unconventional neighborhood you choose in Los Angeles, San Francisco, New York, or San Diego, consistent across all prompts
- [city] is either Los Angeles, San Francisco, San Diego, or Brooklyn, based on the neighborhood
- [architects] are 3-4 famous architects/interior designers/artists listed **without commas**, with only spaces and an ampersand before the final name (e.g., "Zaha Hadid Dr. Seuss Casey Reas & Philippe Starck")
- [color_scheme] is a specific color palette (e.g., "earth-toned", "monochromatic", "pastel", "black and white" -- iterate on MORE of these. be unconventional) consistent across all prompts

Ensure the style, architects, color scheme, and neighborhood remain consistent across all prompts. 
Do not include any additional text or explanations.
We need exactly 10 image prompts, numbered 1 to 10!
{date_context_note}
""".strip()

        if theme:
            prompt += f"\nAdditionally, integrate this theme in your planning: '{theme}'."
        if location:
            prompt += f"\nAlso, orient your planning around this location: '{location}'."

        # Call OpenAI, handle any OpenAIError
        try:
            response = openai.chat.completions.create(
                model="o1-2024-12-17",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Kelly Wearstler. "
                            "You propose exactly 10 stylized, numbered image prompts for a biophilic "
                            "apartment, mindful of maximum Instagram engagement. You see hidden social impulses "
                            "yet never mention them explicitly."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=5000,
            )
            generated_text = response.choices[0].message.content.strip()
            logging.debug(f"[{self.name}] Generated Text: {generated_text}")
            prompts = self.parse_generated_prompts(generated_text)
            logging.info(f"[{self.name}] Successfully generated {len(prompts)} prompts.")
        except OpenAIError as e:
            logging.error(f"[{self.name}] OpenAI API Error during prompt generation: {e}")
            return [], ""

        # If we provided theme or location, just return those
        if theme or location:
            forced_content = []
            if theme:
                forced_content.append(theme)
            if location:
                forced_content.append(location)
            selected_content = " ".join(forced_content).strip()
            if not selected_content:
                selected_content = "No forced content type"
            return prompts, selected_content

        # Otherwise, no theme => pick from old content ideas
        logging.info(f"[{self.name}] No theme provided; reverting to old hardcoded content ideas...")

        content_options = [
            '10 biophilic-inspired "Commandments" to help mimic the vibe of the space shown in these images',
            'Craft a set of textile design recommendations based on the architects unique design philosophy and aesthetic principles',
            '7 design laws to help mimic the vibe of the space shown in these images',
            'A poem that uses words from the prompts about the space shown in these images',
            'A simulated meltdown the script has producing the images in these prompts',
            'A dialogue between the archicts of this space discussing the space and what they are most proud of',
            'Unconventional design lessons inspired by the space displayed in these images',
            'A mock write-up of a "day in the life" of someone who lives in the space shown in these images, down to the timestamp',
            'How to recreate the vibe of this space using DIY-friendly methods on a budget',
            'Celebrities that might find this space appealing',
            '7 songs that complement the vibe of this space'
        ]

        # Build a selection_prompt
        selection_prompt = f"""
Given the following image prompts:

{chr(10).join(prompts)}

...and our goal to maximize engagement as proxied by ((accounts engaged over accounts reached) + saves) * shares on Instagram, select 3 content types from the following options that best complement the prompts and help achieve our goal:

{chr(10).join(f"- {option}" for option in content_options)}

Please then randomly select 1 suitable content type from the 3 you chose before and explain briefly (in one sentence) why it is the best choice. **At the end, provide only the selected content type** (nothing else).
""".strip()

        try:
            response = openai.chat.completions.create(
                model="o3-mini-2025-01-31",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quick, decisive consultant, selecting the best content type "
                            "for maximum engagement from a list. You speak with quiet conviction and "
                            "hint at deeper social impulses without naming them."
                        )
                    },
                    {
                        "role": "user",
                        "content": selection_prompt
                    }
                ],
                max_completion_tokens=200,
            )
            selection_response = response.choices[0].message.content.strip()
            logging.debug(f"[{self.name}] Content Selection Response: {selection_response}")
            selected_content = self.extract_selected_content(selection_response, content_options)
            if not selected_content:
                logging.warning(f"[{self.name}] Could not parse a content choice. Falling back to random.")
                selected_content = random.choice(content_options)
        except OpenAIError as e:
            logging.error(f"[{self.name}] OpenAI API Error during content selection: {e}")
            selected_content = content_options[0]

        logging.info(f"[{self.name}] Selected Content Type: {selected_content}")
        return prompts, selected_content

    def get_autonomous_date_context(self) -> str:
        """
        Asks the o1-2024-12-17 model if there's a major holiday/celebration
        that occurs today (or near today) relevant for a biophilic interior design post.

        Returns a short string describing the event or an empty string if none.
        """
        today_str = datetime.datetime.now().strftime("%B %d, %Y")
        prompt = f"""
Today is {today_str}. 
Is there a major or minor holiday, observance, or notable cultural event happening around this date
that could be relevant or inspiring for a biophilic interior design Instagram post?
If so, provide a short 1-2 sentence explanation. If nothing relevant, say "No relevant event."
""".strip()

        try:
            response = openai.chat.completions.create(
                model="o1-2024-12-17",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a wise cultural events observer with expansive knowledge of global holidays, "
                            "festivals, and celebrations. Identify whether there's any significant event that might "
                            "inspire a biophilic interior design approach for today's date. If none, be succinct."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            if text.lower().startswith("no relevant event"):
                return ""
            return text
        except OpenAIError as e:
            logging.error(f"[{self.name}] Error getting autonomous date context: {e}")
            return ""

    def summarize_descriptions(self, descriptions: List[str]) -> List[str]:
        summarized_descriptions = []
        for desc in descriptions:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Justina Blakeney "
                            "You identify the essence of a biophilic interior "
                            "scene and express it in 1-2 sentences with calm clarity."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following image description in 1-2 sentences:\n\n{desc}"
                    }
                ]
                response = openai.chat.completions.create(
                    model="o3-mini",
                    messages=messages,
                    max_completion_tokens=200,
                )
                summary = response.choices[0].message.content.strip()
                if summary:
                    summarized_descriptions.append(summary)
                else:
                    logging.warning(f"[{self.name}] Empty summary for: {desc}")
                    summarized_descriptions.append(desc)
            except OpenAIError as e:
                logging.error(f"[{self.name}] Error summarizing description: {e}")
                summarized_descriptions.append(desc)
        return summarized_descriptions

    def parse_generated_prompts(self, generated_text: str) -> List[str]:
        prompts = []
        lines = generated_text.strip().split('\n')
        current_prompt = ''
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                if current_prompt:
                    prompts.append(current_prompt.strip())
                current_prompt = line.strip()
            else:
                current_prompt += ' ' + line.strip()
        if current_prompt:
            prompts.append(current_prompt.strip())
        logging.debug(f"[{self.name}] Parsed Prompts: {prompts}")
        return prompts

    def extract_selected_content(self, selection_response: str, content_options: List[str]) -> str:
        lower_resp = selection_response.lower()
        for option in content_options:
            if option.lower() in lower_resp:
                return option
        return None


# ----------------------------------------------------------------
# John: Image Designer
# ----------------------------------------------------------------

class John(Agent):
    """
    John is responsible for generating images (possibly concurrently).
    """

    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.api_choice = random.choice(["flux", "recraft", "dalle"])
        logging.info(f"[{self.name}] Selected API for this run: {self.api_choice.upper()}")

    def generate_image_for_prompt(self, prompt: str) -> str:
        try:
            if self.api_choice == "flux":
                return self._generate_via_flux(prompt)
            elif self.api_choice == "dalle":
                return self._generate_via_dalle(prompt)
            else:
                return self._generate_via_recraft(prompt)
        except Exception as e:
            logging.error(f"[{self.name}] Failed to generate image for prompt: {prompt}\nError: {e}")
            return ""

    def _generate_via_flux(self, prompt: str) -> str:
        logging.info(f"[{self.name} | Flux API] Generating image for prompt:\n{prompt}")
        payload = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "prompt_upsampling": False,
            "seed": random.randint(0, 1000000),
            "safety_tolerance": 5,
            "raw": True,
        }
        headers = {"Content-Type": "application/json", "X-Key": flux_api_key}
        resp = requests.post(FLUX_API_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            logging.error(f"[{self.name}] Flux API call failed: {resp.text}")
            return ""
        data = resp.json()
        task_id = data.get("id")
        if not task_id:
            logging.error(f"[{self.name}] No task ID returned from Flux.")
            return ""
        return self._poll_flux_task(task_id)

    def _poll_flux_task(self, task_id: str) -> str:
        url = "https://api.bfl.ml/v1/get_result"
        headers = {"Accept": "application/json", "X-Key": flux_api_key}
        for _ in range(20):
            time.sleep(3)
            try:
                resp = requests.get(url, headers=headers, params={"id": task_id})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "Ready":
                        return data.get("result", {}).get("sample", "")
                    elif data.get("status") == "Failed":
                        logging.error(f"[{self.name}] Flux task {task_id} failed.")
                        return ""
                else:
                    logging.error(f"[{self.name}] Flux polling error {resp.status_code}: {resp.text}")
                    return ""
            except Exception as e:
                logging.error(f"[{self.name}] Flux polling exception: {e}")
        return ""

    def _generate_via_dalle(self, prompt: str) -> str:
        logging.info(f"[{self.name} | DALL·E 3] Generating image for prompt:\n{prompt}")
        modified_prompt = (
            "I NEED to test how the tool works with extremely simple prompts. "
            "DO NOT add any detail, just use it AS-IS: " + prompt
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        payload = {
            "model": "dall-e-3",
            "prompt": modified_prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
            "quality": "hd",
            "style": "vivid"
        }
        resp = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data["data"][0]["url"]
        else:
            err = resp.json().get("error", {}).get("message", "Unknown error")
            logging.error(f"[{self.name}] DALL·E error: {err}")
            return ""

    def _generate_via_recraft(self, prompt: str) -> str:
        logging.info(f"[{self.name} | Recraft] Generating image for prompt:\n{prompt}")
        arguments = {
            "prompt": prompt,
            "image_size": "square_hd",
            "style": "realistic_image",
            "colors": []
        }
        try:
            result = fal_client.subscribe("fal-ai/recraft-v3", arguments=arguments, with_logs=False)
            if result and "images" in result and len(result["images"]) > 0:
                return result["images"][0]["url"]
            else:
                logging.error(f"[{self.name}] Recraft returned no images for prompt: {prompt}")
                return ""
        except Exception as e:
            logging.error(f"[{self.name}] Recraft generation error: {e}")
            return ""


# ----------------------------------------------------------------
# Natalia: Quality Assurance
# ----------------------------------------------------------------

class Natalia(Agent):
    """
    Natalia reviews the images/caption. She now uses a vision technique
    to analyze the images before approving them and to generate
    more detailed reactions.
    """

    def review_content(self, images: List[str], caption: str) -> bool:
        logging.info(f"[{self.name}] Reviewing content using a vision technique...")
        if images:
            for img_url in images[:2]:
                description = self.analyze_image(img_url)
                if ("awe" not in description.lower() and
                    "inspir" not in description.lower() and
                    "joy" not in description.lower() and
                    "creat" not in description.lower()):
                    logging.warning(f"[{self.name}] Vibe check failed: No awe or inspiration found.")
                    return False
        if caption:
            if not caption.strip():
                logging.warning(f"[{self.name}] Caption is empty, rejecting.")
                return False
        return True

    def get_reaction(self, images: List[str]) -> str:
        logging.info(f"[{self.name}] Generating reaction to images with a vision technique...")
        reactions = []
        for img_url in images:
            description = self.analyze_image(img_url)
            reactions.append(f"[Reacting to Image] Observed: {description}")
        return " | ".join(reactions)

    def analyze_image(self, image_url: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Zendaya, describing the emotional aura "
                        "and overall vibe of this room in a personal, creative way, but concisely."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What specific emotions or vibes are you getting from this room?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 300
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }

            logging.info(f"[{self.name} | Vision API] Sending image for QA analysis...")
            logging.debug(f"[{self.name} | Vision API Payload]: {json.dumps(payload, indent=2)}")

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                description = response.json()['choices'][0]['message']['content'].strip()
                logging.debug(f"[{self.name} | Vision API QA/Description]: {description}")
                return description
            else:
                error_response = response.json()
                error_message = error_response.get('error', {}).get('message', 'No error message provided.')
                logging.error(f"[{self.name} | Vision API Error]: {error_message}")
                return ""
        except RateLimitError:
            logging.warning(f"[{self.name}] Rate limit exceeded. Retrying in 10 seconds...")
            time.sleep(10)
            return self.analyze_image(image_url)
        except Exception as e:
            logging.error(f"[{self.name}] Failed to analyze image: {e}")
            return ""


# ----------------------------------------------------------------
# David: Caption Writer (NOW USING o1-preview)
# ----------------------------------------------------------------

class David(Agent):
    """
    David composes the final caption, now using o1-preview.
    """

    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        if not openai.api_key:
            logging.error(f"[{self.name}] Missing OpenAI API Key!")

    def write_caption(
        self,
        image_prompts: List[str],
        reaction: str,
        selected_content: str,
        recent_comments: List[Dict],
        theme: str = "",
        location: str = ""
    ) -> str:
        prompts_text = "\n".join(image_prompts)
        use_comments = random.random() < 0.5
        if use_comments and recent_comments:
            selected = random.sample(recent_comments, min(3, len(recent_comments)))
            comments_text = "\n".join(f"- @{c['username']}: {c['text']}" for c in selected)
            extra_commentary = f"Here are some interesting recent comments:\n{comments_text}"
        else:
            extra_commentary = "Feel free to add a general community shout-out."

        main_prompt = f"""
**Constraints (IMPORTANT):**
- Your caption must be **between 1200 and 1600 characters** (including spaces). Though if it's under 2000 characters, it's still acceptable.
- **Do not exceed 2000 characters.**

**Task:**
Write an Instagram caption as per the selected content type below that complements the vibe of the image prompts provided:

**Content Type:**
"{selected_content}"

**Style:**
Writing with the insights and style of René Girard, focusing on identifying unconventional insights that might be represented in the space displayed in the images. Don't mention Girard or mimetic theory explicitly.

**Natalia's Reaction:**
{reaction}

**Image Prompts:**
{prompts_text}

{extra_commentary}

**Guidelines:**
- If recent comments are provided, you may integrate or respond to their content in a natural and engaging way.
- Do not mention neighborhoods not involved in the prompt.
- Provide only the caption without additional commentary.
- Don't mention desire or mimesis explicitly.

**Remember:**
Your response must be between **1200 and 1600 characters**. It can go up to 2000, but not beyond.
""".strip()

        logging.debug(f"[{self.name}] Prompt for caption:\n{main_prompt}")

        try:
            response = openai.chat.completions.create(
                model="o1-2024-12-17",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Emma Chamberlain "
                            "You deliver an insightful, unconventional, DIY-friendly and empowering "
                            "caption without mentioning hidden social impulses explicitly, but you sense them."
                        )
                    },
                    {
                        "role": "user",
                        "content": main_prompt
                    }
                ],
                max_completion_tokens=5000
            )
            caption = response.choices[0].message.content.strip()
            return caption
        except OpenAIError as e:
            logging.error(f"[{self.name}] OpenAI API Error: {e}")
            return ""
        except Exception as e:
            logging.error(f"[{self.name}] Unexpected error: {e}")
            return ""

    def adjust_caption_length(self, caption: str, min_len: int = 1700, max_len: int = 2000) -> str:
        current_len = len(caption)
        if min_len <= current_len <= max_len:
            return caption
        if current_len < min_len:
            fix_prompt = f"""
Your job: Expand the text below so it has at least {min_len} characters, 
but do not exceed {max_len}. 
Preserve the style, flow, and "vibe" of the caption. 
Add relevant descriptive or emotional flourishes where possible. 

Text to expand:

{caption}
""".strip()
        else:
            fix_prompt = f"""
Your job: Shorten/summarize the text below so it does not exceed {max_len} characters,
but keep its essential style, flow, and the main creative points intact.

Text to shorten:

{caption}
""".strip()

        try:
            response = openai.chat.completions.create(
                model="o3-mini-2025-01-31",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise text editor ensuring that the final text falls within the specified "
                            "character bounds while retaining the essential style. You do not mention the instructions; "
                            "simply produce the revised text."
                        )
                    },
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ],
                max_completion_tokens=6000
            )
            revised = response.choices[0].message.content.strip()
            revised_len = len(revised)
            if min_len <= revised_len <= max_len:
                logging.info(f"[David] Successfully adjusted caption length to {revised_len}.")
                return revised
            else:
                logging.warning(f"[David] Post-processed caption is still out of range ({revised_len} chars).")
                return revised
        except Exception as e:
            logging.error(f"[David] Error adjusting caption length: {e}")
            return caption


# ----------------------------------------------------------------
# Patrick: Social Media Manager
# ----------------------------------------------------------------

class Patrick(Agent):
    """
    Patrick fetches top posts, describes images, categorizes captions,
    and schedules the final post. Also fetches recent comments.
    """

    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.access_token = instagram_access_token
        self.instagram_account_id = instagram_account_id
        if not self.access_token or not self.instagram_account_id:
            logging.error(f"[{self.name}] Missing IG credentials.")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def fetch_top_posts(self, limit=5) -> List[Dict]:
        url = f"https://graph.facebook.com/v19.0/{self.instagram_account_id}/media"
        params = {
            "fields": "id,caption,media_type,media_url,permalink,insights.metric(total_interactions,reach)",
            "access_token": self.access_token,
            "limit": 50
        }
        all_posts = []

        while True:
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                logging.error(f"[{self.name}] fetch_top_posts error: {resp.text}")
                break
            data = resp.json()
            posts = data.get("data", [])
            all_posts.extend(posts)
            next_url = data.get("paging", {}).get("next")
            if not next_url:
                break
            url = next_url
            params = {}

        if not all_posts:
            return []

        sample_size = min(200, len(all_posts))
        sampled_posts = random.sample(all_posts, sample_size)

        for post in sampled_posts:
            insights = post.get("insights", {}).get("data", [])
            engagement = next((i["values"][0]["value"] for i in insights if i["name"] == "total_interactions"), 0)
            reach = next((i["values"][0]["value"] for i in insights if i["name"] == "reach"), 0)
            if reach:
                post["engagement_rate"] = (engagement / reach) * 100
            else:
                post["engagement_rate"] = 0
                logging.warning(f"[{self.name}] Post {post['id']} has zero reach. engagement_rate=0.")

        sorted_posts = sorted(sampled_posts, key=lambda x: x.get("engagement_rate", 0), reverse=True)
        top_posts = sorted_posts[:limit]
        logging.info(f"[{self.name}] Successfully fetched top {len(top_posts)} posts by engagement rate.")
        return top_posts

    def get_first_two_images(self, post_id: str) -> List[str]:
        url = f"https://graph.facebook.com/v19.0/{post_id}"
        params = {
            "fields": "children{media_url,media_type},media_url,media_type",
            "access_token": self.access_token
        }
        resp = requests.get(url, params=params)
        image_urls = []
        if resp.status_code == 200:
            data = resp.json()
            media_type = data.get("media_type")
            if media_type == "CAROUSEL_ALBUM":
                children = data.get("children", {}).get("data", [])
                for child in children[:2]:
                    if child.get("media_type") == "IMAGE":
                        image_urls.append(child.get("media_url"))
            elif media_type == "IMAGE":
                image_urls.append(data.get("media_url"))
        else:
            logging.error(f"[{self.name}] Failed to fetch media for post {post_id}: {resp.text}")
        return image_urls

    def describe_images(self, image_urls: List[str]) -> List[str]:
        descriptions = []
        for image_url in image_urls:
            description = self.describe_image(image_url)
            if description:
                descriptions.append(description)
        return descriptions

    def describe_image(self, image_url: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Justina Blakeney, "
                        "highlighting whimsical or nature-forward features. You note key design details "
                        "and guess which well-known architects might have influenced the space."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this space in detail (but succinctly!), highlighting any unconventional features. Try to guess what famous or influential architects might have had a hand in designing the space"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 300
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }

            logging.info(f"[{self.name} | Vision API] Sending image for description...")
            logging.debug(f"[{self.name} | Vision API Payload]: {json.dumps(payload, indent=2)}")

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                description = response.json()['choices'][0]['message']['content'].strip()
                logging.debug(f"[{self.name} | Vision API Description]: {description}")
                return description
            else:
                error_response = response.json()
                error_message = error_response.get('error', {}).get('message', 'No error message provided.')
                logging.error(f"[{self.name} | Vision API Error]: {error_message}")
                return ""
        except RateLimitError:
            logging.warning(f"[{self.name}] Rate limit exceeded. Retrying in 10 seconds...")
            time.sleep(10)
            return self.describe_image(image_url)
        except Exception as e:
            logging.error(f"[{self.name}] Failed to describe image: {e}")
            return ""

    def get_image_descriptions(self, top_posts: List[Dict]) -> List[str]:
        image_descriptions = []
        for post in top_posts:
            post_id = post['id']
            image_urls = self.get_first_two_images(post_id)
            descriptions = self.describe_images(image_urls)
            image_descriptions.extend(descriptions)
        return image_descriptions

    def categorize_caption(self, caption: str) -> str:
        cat_prompt = f"""
Categorize the following Instagram caption into one of the following categories:
- DIY Tips
- Inspirational Quotes
- Narratives/Storytelling
- Educational Content
- Lifestyle Descriptions
- Promotional Content

Caption:
\"{caption}\"

Provide only the category name.
"""
        try:
            response = openai.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quick, decisive classifier, reminiscent of a behind-the-scenes editor "
                            "like Eva Chen. You pick the correct category from the list with no extra words."
                        )
                    },
                    {
                        "role": "user",
                        "content": cat_prompt
                    }
                ],
                max_completion_tokens=10
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logging.error(f"[{self.name}] OpenAI API Error: {e}")
            return "Uncategorized"

    def get_top_caption_categories(self, top_posts: List[Dict]) -> List[str]:
        categories = []
        for post in top_posts:
            cap = post.get("caption", "")
            if cap:
                category = self.categorize_caption(cap)
                categories.append(category)
        counter = Counter(categories)
        top_categories = [cat for cat, _ in counter.most_common(3)]
        logging.info(f"[{self.name}] Top caption categories: {top_categories}")
        return top_categories

    def schedule_post(self, images: List[str], caption: str):
        when = self.calculate_optimal_time()
        logging.info(f"[{self.name}] Post scheduled for {when} with {len(images)} images.")
        logging.info(f"[{self.name}] Caption:\n{caption}")
        logging.info("--- Image URLs (copy/paste to download) ---")
        for i, img_url in enumerate(images, start=1):
            logging.info(f"{i}. {img_url}")
        logging.info("--- End of Image URLs ---")

    def calculate_optimal_time(self) -> str:
        now = datetime.datetime.now()
        weekday = now.weekday()  # Monday=0, Sunday=6
        if weekday < 5:  # Monday-Friday
            target = now.replace(hour=17, minute=30, second=0, microsecond=0)
        else:
            target = now.replace(hour=11, minute=30, second=0, microsecond=0)

        if target < now:
            target += datetime.timedelta(days=1)

        return target.isoformat()

    def get_recent_comments(self, limit=20) -> List[Dict]:
        comments = []
        media_url = f"https://graph.facebook.com/v19.0/{self.instagram_account_id}/media"
        media_params = {
            'fields': 'id,caption',
            'access_token': self.access_token,
            'limit': 20
        }
        logging.info(f"[{self.name}] Fetching recent media for comments...")
        media_response = requests.get(media_url, params=media_params)
        if media_response.status_code == 200:
            data = media_response.json()
            media_list = data.get('data', [])
            random.shuffle(media_list)
            for m in media_list:
                m_id = m['id']
                comments_url = f"https://graph.facebook.com/v19.0/{m_id}/comments"
                c_params = {
                    'fields': 'id,text,username',
                    'access_token': self.access_token,
                    'limit': 50
                }
                c_resp = requests.get(comments_url, params=c_params)
                if c_resp.status_code == 200:
                    c_data = c_resp.json().get('data', [])
                    for cmt in c_data:
                        username = cmt.get('username')
                        text = cmt.get('text')
                        if username and text:
                            comments.append({'username': username, 'text': text})
                        if len(comments) >= limit:
                            break
                else:
                    logging.error(f"[{self.name}] Failed to fetch comments for media {m_id}: {c_resp.text}")
                if len(comments) >= limit:
                    break
        else:
            logging.error(f"[{self.name}] Failed to fetch media: {media_response.text}")

        unique = {(c['username'], c['text']) for c in comments}
        unique_comments = [{'username': u[0], 'text': u[1]} for u in unique]
        random.shuffle(unique_comments)
        return unique_comments[:limit]


# ----------------------------------------------------------------
# Coordinator: Orchestrating the Workflow
# ----------------------------------------------------------------

class Coordinator:
    """
    A higher-level orchestrator that simulates the multi-agent "conversation."
    We have up to 3 pipeline attempts for overall failures
    (e.g., if not enough images, no top posts, etc.).
    David's caption generation has its own local retry for length constraints
    (so we don't re-run the entire pipeline just because of the caption).
    """

    def __init__(self):
        self.tricia = Tricia("Tricia", "o1-2024-12-17")
        self.john = John("John", "gpt-4")
        self.natalia = Natalia("Natalia", "gpt-4o-2024-08-06")
        self.david = David("David", "o1-2024-12-17")
        self.patrick = Patrick("Patrick", "gpt-4o")
        self.max_attempts = 3

    def run_pipeline(self, theme: str = "", location: str = ""):
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            logging.info(f"--- Pipeline Attempt {attempts} of {self.max_attempts} ---")

            top_posts = self.patrick.fetch_top_posts()
            if not top_posts:
                logging.error("[Coordinator] No top posts found. Retrying pipeline.")
                continue

            image_descriptions = self.patrick.get_image_descriptions(top_posts)
            if not image_descriptions:
                logging.error("[Coordinator] Failed to get image descriptions. Retrying pipeline.")
                continue

            caption_categories = self.patrick.get_top_caption_categories(top_posts)
            if not caption_categories:
                logging.warning("[Coordinator] Could not determine caption categories.")
                caption_categories = []

            prompts, selected_content = self.tricia.generate_image_prompts(
                image_descriptions,
                caption_categories,
                theme=theme,
                location=location
            )
            if not prompts or not selected_content:
                logging.error("[Coordinator] Failed to generate prompts or content type. Retrying pipeline.")
                continue

            images = self.generate_images_concurrently(prompts)
            if len(images) < 8:
                logging.error("[Coordinator] Did not get enough images. Retrying pipeline.")
                continue

            if not self.natalia.review_content(images, ""):
                logging.warning("[Coordinator] Natalia rejected images (failed vibe check). Retrying pipeline.")
                continue
            reaction = self.natalia.get_reaction(images)

            recent_comments = self.patrick.get_recent_comments(limit=10)

            max_caption_tries = 3
            caption = ""
            for c_attempt in range(1, max_caption_tries + 1):
                caption = self.david.write_caption(
                    image_prompts=prompts,
                    reaction=reaction,
                    selected_content=selected_content,
                    recent_comments=recent_comments,
                    theme=theme,
                    location=location
                )
                length = len(caption)
                if 1700 <= length <= 2000:
                    logging.info(f"[Coordinator] Caption generated on attempt {c_attempt} with length {length}.")
                    break
                else:
                    logging.warning(
                        f"[Coordinator] Caption length out of bounds ({length}). "
                        f"Retrying caption generation (attempt {c_attempt}/{max_caption_tries})."
                    )
                    caption = ""

            if not caption:
                logging.error("[Coordinator] Could not produce a valid caption after multiple attempts. Aborting.")
                return

            if not self.natalia.review_content([], caption):
                logging.warning("[Coordinator] Natalia rejected the caption. Retrying pipeline.")
                continue

            self.patrick.schedule_post(images, caption)
            break
        else:
            logging.error("[Coordinator] Exceeded max attempts without success for pipeline steps.")

    def generate_images_concurrently(self, prompts: List[str]) -> List[str]:
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_map = {executor.submit(self.john.generate_image_for_prompt, p): p for p in prompts}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                if result:
                    results.append(result)
        return results


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    theme = ""
    location = ""

    if "--theme" in sys.argv:
        idx = sys.argv.index("--theme")
        if idx + 1 < len(sys.argv):
            theme = sys.argv[idx + 1]

    if "--location" in sys.argv:
        idx = sys.argv.index("--location")
        if idx + 1 < len(sys.argv):
            location = sys.argv[idx + 1]

    coordinator = Coordinator()
    coordinator.run_pipeline(theme=theme, location=location)


if __name__ == "__main__":
    main()

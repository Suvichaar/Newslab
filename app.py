import json
import time
import os
import uuid
import requests
import boto3
import nltk
import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
import streamlit as st  # ✅ <--- Add this line
from collections import OrderedDict
import zipfile
import io
from datetime import datetime
import random, string, base64   # already present? keep only one copy
from urllib.parse import urlparse
from io import BytesIO
import zipfile
import os, uuid, json, random, string, base64, re, time
from io import BytesIO
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint= st.secrets["azure_api"]["AZURE_OPENAI_ENDPOINT"],
    api_key= st.secrets["azure_api"]["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01"
)

AZURE_TTS_URL = st.secrets["azure"]["AZURE_TTS_URL"]
AZURE_API_KEY = st.secrets["azure"]["AZURE_API_KEY"]

AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION = st.secrets["aws"]["AWS_REGION"]
AWS_BUCKET = st.secrets["aws"]["AWS_BUCKET"]
S3_PREFIX = st.secrets["aws"]["S3_PREFIX"]
CDN_BASE = st.secrets["aws"]["CDN_BASE"]
CDN_PREFIX_MEDIA = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id     = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_KEY,
    region_name           = AWS_REGION,
)


voice_options = {
    "1": "alloy",
    "2": "echo",
    "3": "fable",
    "4": "onyx",
    "5": "nova",
    "6": "shimmer"
}
# Slug and URL generator
def generate_slug_and_urls(title):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    
    slug = ''.join(c for c in title.lower().replace(" ", "-").replace("_", "-") if c in string.ascii_lowercase + string.digits + '-')
    slug = slug.strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}" # this is the urlslug -> slug_nano.html
    return nano, slug_nano, f"https://suvichaar.org/stories/{slug_nano}", f"https://stories.suvichaar.org/{slug_nano}.html"

# === Utility Functions ===
def extract_article(url):
    import newspaper
    from newspaper import Article

    try:
        article = Article(url)
        article.download()
        article.parse()

        try:
            article.nlp()
        except:
            pass  # Some articles may not support NLP extraction

        # Fallbacks for missing fields
        title = article.title or "Untitled Article"
        text = article.text or "No article content available."
        summary = article.summary or text[:300]

        return title.strip(), summary.strip(), text.strip()

    except Exception as e:
        st.error(f"❌ Failed to extract article from URL. Error: {str(e)}")
        return "Untitled Article", "No summary available.", "No article content available."


def get_sentiment(text):
    from textblob import TextBlob

    if not text or not text.strip():
        return "neutral"  # default for empty input

    # Clean and analyze
    clean_text = text.strip().replace("\n", " ")
    polarity = TextBlob(clean_text).sentiment.polarity

    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def detect_category_and_subcategory(text, content_language="English"):
    import json

    if not text or len(text.strip()) < 50:
        return {
            "category": "Unknown",
            "subcategory": "General",
            "emotion": "Neutral"
        }

    # Prompt construction based on language
    if content_language == "Hindi":
        prompt = f"""
आप एक समाचार विश्लेषण विशेषज्ञ हैं।

इस समाचार लेख का विश्लेषण करें और नीचे तीन बातें बताएं:

1. category (श्रेणी)
2. subcategory (उपश्रेणी)
3. emotion (भावना)

लेख:
\"\"\"{text[:3000]}\"\"\"

जवाब केवल JSON में दें:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""
    else:
        prompt = f"""
You are an expert news analyst.

Analyze the following news article and return:

1. category
2. subcategory
3. emotion

Article:
\"\"\"{text[:3000]}\"\"\"

Return ONLY as JSON:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify the news into category, subcategory, and emotion."},
                {"role": "user", "content": prompt.strip()}
            ],
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        content = content.strip("```json").strip("```").strip()

        result = json.loads(content)

        if all(k in result for k in ["category", "subcategory", "emotion"]):
            return result

    except Exception as e:
        print("❌ Category detection failed:", e)

    return {
        "category": "Unknown",
        "subcategory": "General",
        "emotion": "Neutral"
    }


def title_script_generator(category, subcategory, emotion, article_text, content_language="English", character_sketch=None):
    if not character_sketch:
        character_sketch = (
            f"Polaris is a sincere and articulate {content_language} news anchor. "
            "They present facts clearly, concisely, and warmly, connecting deeply with their audience."
        )

    # 🔹 Prompt to generate slides (excluding slide 1 narration)
    system_prompt = f"""
You are a digital content editor.

Create a structured 5-slide web story from the article below.

Language: {content_language}

Each slide must contain:
- A short title in {content_language}
{"- The title must be written in Hindi (Devanagari script)." if content_language == "Hindi" else ""}
- A narration prompt (instruction only, don't write narration)
{"- The narration prompt must also be in Hindi (Devanagari script)." if content_language == "Hindi" else ""}

Format:
{{
  "slides": [
    {{ "title": "...", "prompt": "..." }},
    ...
  ]
}}
"""

    user_prompt = f"""
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Article:
\"\"\"{article_text[:3000]}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.strip("```json").strip("```").strip()

    try:
        slides_raw = json.loads(content)["slides"]
    except:
        return {"category": category, "subcategory": subcategory, "emotion": emotion, "slides": []}

    # 🔹 Generate Slide 1 Intro Narration
    headline = article_text.split("\n")[0].strip().replace('"', '')

    if content_language == "Hindi":
        slide1_prompt = f"Generate a greeting and news headline narration in Hindi for the story: {headline}"
    else:
        slide1_prompt = f"Generate a greeting and headline intro narration in English for: {headline}"

    slide1_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a news presenter generating opening lines."},
            {"role": "user", "content": slide1_prompt}
        ]
    )
    slide1_script = slide1_response.choices[0].message.content.strip()

    slides = [{
        "title": headline[:80],
        "prompt": "Intro slide with greeting and headline.",
        "image_prompt": f"Vector-style illustration of Polaris presenting news: {headline}",
        "script": slide1_script
    }]

    # 🔹 Generate narration for each slide
    for slide in slides_raw:
        script_language = f"{content_language} (use Devanagari script)" if content_language == "Hindi" else content_language
        narration_prompt = f"""
Write a narration in **{script_language}** (max 200 characters),
in the voice of Polaris.

Instruction: {slide['prompt']}
Tone: Warm, clear, informative. No self-intro.

Character sketch:
{character_sketch}
"""

        try:
            narration_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You write concise narrations for web story slides."},
                    {"role": "user", "content": narration_prompt.strip()}
                ]
            )
            narration = narration_response.choices[0].message.content.strip()
        except:
            narration = "Unable to generate narration for this slide."

        slides.append({
            "title": slide['title'],
            "prompt": slide['prompt'],
            "image_prompt": f"Modern vector-style visual for: {slide['title']}",
            "script": narration
        })

    return {
        "category": category,
        "subcategory": subcategory,
        "emotion": emotion,
        "slides": slides
    }



def modify_tab4_json(original_json):
    updated_json = OrderedDict()
    slide_number = 2  # Start from slide2 since slide1 & slide2 are removed

    for i in range(3, 100):  # Covers slide3 to slide99
        old_key = f"slide{i}"
        if old_key not in original_json:
            break
        content = original_json[old_key]
        new_key = f"slide{slide_number}"

        for k, v in content.items():
            if k.endswith("paragraph1"):
                para_key = f"s{slide_number}paragraph1"
                audio_key = f"audio_url{slide_number}"
                updated_json[new_key] = {
                    para_key: v,
                    audio_key: content.get("audio_url", ""),
                    "voice": content.get("voice", "")
                }
                break
        slide_number += 1

    return updated_json

def replace_placeholders_in_html(html_text, json_data):
    storytitle = json_data.get("slide1", {}).get("storytitle", "")
    storytitle_url = json_data.get("slide1", {}).get("audio_url", "")
    hookline = json_data.get("slide2", {}).get("hookline", "")
    hookline_url = json_data.get("slide2", {}).get("audio_url", "")

    html_text = html_text.replace("{{storytitle}}", storytitle)
    html_text = html_text.replace("{{storytitle_audiourl}}", storytitle_url)
    html_text = html_text.replace("{{hookline}}", hookline)
    html_text = html_text.replace("{{hookline_audiourl}}", hookline_url)

    return html_text

# Tab 4 layout // Hookline modified 
def generate_hookline(title, summary, content_language="English"):
    # Prepare prompt based on language
    if content_language == "Hindi":
        prompt = f"""
आप 'पोलारिस' नामक एक सोशल मीडिया रणनीतिकार हैं और चैनल 'सुविचार' के लिए कार्य करते हैं। आपका कार्य है एक संक्षिप्त, ध्यान खींचने वाली *हुकलाइन* बनाना जो इस समाचार की ओर दर्शकों का ध्यान आकर्षित करे।

शीर्षक: {title}
सारांश: {summary}

भाषा: हिंदी

अनुरोध:
- केवल एक वाक्य हो
- हैशटैग, इमोजी या अधिक विराम चिह्न न हो
- भाषा सरल और भावनात्मक रूप से आकर्षक हो
- 120 वर्णों से कम होनी चाहिए
- उद्धरण ("") शामिल न करें

उदाहरण:
- सरकार का यह कदम सबको चौंका देगा।
- भारत का अंतरिक्ष में साहसिक कदम।

अब कृपया पोलारिस की आवाज़ में हुकलाइन दीजिए:
"""
    else:
        prompt = f"""
You are Polaris, a social media strategist for the news channel 'Suvichaar'. Your job is to create a short, attention-grabbing *hookline* for a news story.

Title: {title}
Summary: {summary}

Language: {content_language}

Requirements:
- One sentence only
- Avoid hashtags, emojis, and excessive punctuation
- Use simple and emotionally engaging language
- Must be under 120 characters
- Do not include quotes in output

Example formats:
- What happened next will shock you.
- India's bold step in space tech.

Now generate the hookline in Polaris' tone:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You create viral hooklines for news stories."},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        return response.choices[0].message.content.strip().strip('"')

    except Exception as e:
        print(f"❌ Hookline generation failed: {e}")
        return "यह खबर आपको चौंका सकती है!" if content_language == "Hindi" else "This story might surprise you!"




def restructure_slide_output(final_output):
    slides = final_output.get("slides", [])
    structured = {}

    for idx, slide in enumerate(slides):
        key = f"s{idx + 1}paragraph1"
        script = slide.get("script", "").strip()

        # Safety net: If empty script, fall back to title or prompt
        if not script:
            fallback = slide.get("title") or slide.get("prompt") or "Content unavailable"
            script = fallback.strip()

        structured[key] = script

    return structured

def generate_remotion_input(tts_output: dict, fixed_image_url: str, author_name: str = "Suvichaar"):
    remotion_data = OrderedDict()
    slide_index = 1

    # Slide 1: storytitle
    if "storytitle" in tts_output:
        remotion_data[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": tts_output["storytitle"],
            f"s{slide_index}audio1": tts_output.get(f"slide{slide_index}", {}).get("audio_url", ""),
            f"s{slide_index}image1": fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # Slides for s1paragraph1 to s9paragraph1
    for i in range(1, 10):
        key = f"s{i}paragraph1"
        if key in tts_output:
            slide_key = f"slide{slide_index}"
            remotion_data[slide_key] = {
                f"s{slide_index}paragraph1": tts_output[key],
                f"s{slide_index}audio1": tts_output.get(slide_key, {}).get("audio_url", ""),
                f"s{slide_index}image1": fixed_image_url,
                f"s{slide_index}paragraph2": f"- {author_name}"
            }
            slide_index += 1

    # Hookline as last content slide
    if "hookline" in tts_output:
        slide_key = f"slide{slide_index}"
        remotion_data[slide_key] = {
            f"s{slide_index}paragraph1": tts_output["hookline"],
            f"s{slide_index}audio1": tts_output.get(slide_key, {}).get("audio_url", ""),
            f"s{slide_index}image1": fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # ✅ Final CTA slide
    remotion_data[f"slide{slide_index}"] = {
        f"s{slide_index}paragraph1": "Get Such\nInspirational\nContent",
        f"s{slide_index}audio1": "https://cdn.suvichaar.org/media/tts_407078a4ff494fb5bed8c35050ffd1a7.mp3",
        f"s{slide_index}video1": "",
        f"s{slide_index}paragraph2": "Like | Subscribe | Share\nwww.suvichaar.org"
    }

    # Save to file
    timestamp = int(time.time())
    filename = f"remotion_input_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(remotion_data, f, indent=2, ensure_ascii=False)

    return filename



def synthesize_and_upload(paragraphs, voice):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    result = OrderedDict()
    os.makedirs("temp", exist_ok=True)

    slide_index = 1

    # Slide 1: storytitle
    if "storytitle" in paragraphs:
        storytitle = paragraphs["storytitle"]
        response = requests.post(
            AZURE_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_API_KEY
            },
            json={
                "model": "tts-1-hd",
                "input": storytitle,
                "voice": voice
            }
        )
        response.raise_for_status()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(response.content)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"
        result[f"slide{slide_index}"] = {
            "storytitle": storytitle,
            "audio_url": cdn_url,
            "voice": voice
        }
        os.remove(local_path)
        slide_index += 1

    # Slide 2: hookline
    if "hookline" in paragraphs:
        hookline = paragraphs["hookline"]
        response = requests.post(
            AZURE_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_API_KEY
            },
            json={
                "model": "tts-1-hd",
                "input": hookline,
                "voice": voice
            }
        )
        response.raise_for_status()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(response.content)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"
        result[f"slide{slide_index}"] = {
            "hookline": hookline,
            "audio_url": cdn_url,
            "voice": voice
        }
        os.remove(local_path)
        slide_index += 1

    # Slide 3 onwards: s1paragraph1 to s9paragraph1
    for i in range(1, 10):  # s1 to s9
        key = f"s{i}paragraph1"
        if key not in paragraphs:
            continue

        text = paragraphs[key]
        st.write(f"🛠️ Processing {key}")

        response = requests.post(
            AZURE_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_API_KEY
            },
            json={
                "model": "tts-1-hd",
                "input": text,
                "voice": voice
            }
        )
        response.raise_for_status()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(response.content)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"

        result[f"slide{slide_index}"] = {
            key: text,
            "audio_url": cdn_url,
            "voice": voice
        }

        os.remove(local_path)
        slide_index += 1

    return result

def transliterate_to_devanagari(json_data):
    updated = {}

    for k, v in json_data.items():
        # Only transliterate slide paragraphs
        if k.startswith("s") and "paragraph1" in k and v.strip():
            prompt = f"""Transliterate this Hindi sentence (written in Latin script) into Hindi Devanagari script. Return only the transliterated text:\n\n{v}"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a Hindi transliteration expert."},
                        {"role": "user", "content": prompt.strip()}
                    ]
                )
                devanagari = response.choices[0].message.content.strip()
                updated[k] = devanagari
            except Exception as e:
                # Fallback: use original if error occurs
                updated[k] = v
        else:
            updated[k] = v

    return updated

def generate_storytitle(title, summary, content_language="English"):
    if content_language == "Hindi":
        prompt = f"""
आप एक समाचार शीर्षक विशेषज्ञ हैं। नीचे दी गई अंग्रेज़ी समाचार शीर्षक और सारांश को पढ़कर, उसी का अर्थ बनाए रखते हुए एक नया आकर्षक **हिंदी शीर्षक** बनाइए।

अंग्रेज़ी शीर्षक: {title}
सारांश: {summary}

अनुरोध:
- केवल एक पंक्ति
- भाषा सरल और स्पष्ट हो
- भावनात्मक रूप से आकर्षक हो
- उद्धरण ("") शामिल न करें

अब कृपया हिंदी शीर्षक दीजिए:
"""
    else:
        return title.strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You generate clear and catchy news headlines."},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        return response.choices[0].message.content.strip().strip('"')

    except Exception as e:
        print(f"❌ Storytitle generation failed: {e}")
        return title.strip()


# === Streamlit UI ===
st.title("🧠 Web Story Content Generator")

tab1, tab2, tab3 ,tab4 ,tab5 ,tab6 , = st.tabs(["Step:1", "Step:2", "Step:3","Step:4","Step:5","Step:6"])

# 🧠 Streamlit UI – Tab 1
with tab1:
    st.title("🧠 Generalized Web Story Prompt Generator")

    url = st.text_input("Enter a news article URL")
    persona = st.selectbox(
        "Choose audience persona:",
        ["genz", "millenial", "working professionals", "creative thinkers", "spiritual explorers"]
    )
    content_language = st.selectbox("Choose content language", ["English", "Hindi"])
    number = st.number_input(
        "Enter no of slides except (Storytitle and Hookline)",
        min_value=0,
        max_value=1000,
        value=10,
        step=1
    )

    if st.button("🚀 Submit and Generate JSON"):
        if url and persona:
            with st.spinner("Analyzing the article and generating prompts..."):
                try:
                    # Step 1–5: Extract + Analyze
                    title, summary, full_text = extract_article(url)
                    sentiment = get_sentiment(summary or full_text)
                    result = detect_category_and_subcategory(full_text)
                    category = result["category"]
                    subcategory = result["subcategory"]
                    emotion = result["emotion"]

                    # Step 6: Generate hookline and storytitle with language handling
                    hookline = generate_hookline(title, summary, content_language)
                    storytitle = generate_storytitle(title, summary, content_language)

                    # Step 7: Generate slide content
                    output = title_script_generator(
                        category, subcategory, emotion, full_text, content_language
                    )

                    final_output = {
                        "title": title,
                        "summary": summary,
                        "sentiment": sentiment,
                        "emotion": emotion,
                        "category": category,
                        "subcategory": subcategory,
                        "persona": persona,
                        "slides": output.get("slides", []),
                        "storytitle": storytitle,
                        "hookline": hookline
                    }

                    # Step 8: Flatten into story JSON
                    structured_output = OrderedDict()
                    structured_output["storytitle"] = storytitle

                    for i in range(1, number + 1):
                        key = f"s{i}paragraph1"
                        structured_output[key] = restructure_slide_output(final_output).get(key, "")

                    structured_output["hookline"] = hookline

                    # Step 9: Hindi transliteration (only for paragraphs, not title/hookline)
                    if content_language == "Hindi":
                        structured_output = transliterate_to_devanagari(structured_output)

                    # Step 10: Save + Download JSON
                    timestamp = int(time.time())
                    filename = f"structured_slides_{timestamp}.json"

                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(structured_output, f, indent=2, ensure_ascii=False)

                    with open(filename, "r", encoding="utf-8") as f:
                        st.success("✅ Prompt generation complete!! Click below to download:")
                        st.download_button(
                            label=f"⬇️ Download JSON ({timestamp})",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a valid URL and choose a persona.")

with tab2:
    st.title("🎙️ GPT-4o Text-to-Speech to S3")
    uploaded_file = st.file_uploader("Upload structured slide JSON", type=["json"])
    voice_label = st.selectbox("Choose Voice", list(voice_options.values()))

    if uploaded_file and voice_label:
        paragraphs = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(paragraphs)} paragraphs")

        if st.button("🚀 Generate TTS + Upload to S3"):
            
            with st.spinner("Please wait..."):
                output = synthesize_and_upload(paragraphs, voice_label)
                st.success("✅ Done uploading to S3!")
                timestamp = int(time.time())
                output_filename = f"tts_output_{timestamp}.json"
        
                # Save TTS output
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
        
                # ✅ Remotion generation
                fixed_image_url = "https://media.suvichaar.org/upload/polaris/polariscover.png"
                remotion_filename = generate_remotion_input(output, fixed_image_url, author_name="Suvichaar")
        
                # 📥 Download TTS JSON
                with open(output_filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download Output JSON",
                        data=f.read(),
                        file_name=output_filename,
                        mime="application/json"
                    )
                    
with tab3:

    st.title("🧩Saving modified file")

    uploaded_file = st.file_uploader("📤 Upload Full Slide JSON (with slide1 to slide8)", type=["json"])

if uploaded_file:
    json_data = json.load(uploaded_file)
    st.success("✅ JSON Loaded")

    try:
        with open("test.html", "r", encoding="utf-8") as f:
            html_template = f.read()
    except FileNotFoundError:
        st.error("❌ Could not find `templates/test.html`. Please make sure it exists.")
    else:
        updated_html = replace_placeholders_in_html(html_template, json_data)
        updated_json = modify_tab4_json(json_data)

        if st.button("🎯 Generate Final HTML + Trimmed JSON (ZIP)"):
            # generate timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"Output_bundle_{ts}.zip"

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                # you can also include the timestamp in the internal filenames if desired:
                zipf.writestr(f"updated_test_{ts}.html", updated_html)
                zipf.writestr(f"output_{ts}.json", json.dumps(updated_json, indent=2, ensure_ascii=False))
            buffer.seek(0)

            st.download_button(
                label="⬇️ Download ZIP with HTML + JSON",
                data=buffer,
                file_name=zip_filename,
                mime="application/zip"
            )

with tab4:
    #
    # Streamlit UI
    st.title("🎞️ AMP Web Story Generator with Full Animation and Audio")
    
    # Fixed path to the AMP HTML template file
    TEMPLATE_PATH = Path("test.html")
    
    # Function to generate an AMP slide using paragraph and audio URL
    def generate_slide(paragraph: str, audio_url: str):
        return f"""
        <amp-story-page id="c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a" auto-advance-after="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" class="i-amphtml-layout-container" i-amphtml-layout="container">
            <amp-story-animation layout="nodisplay" trigger="visibility" class="i-amphtml-layout-nodisplay" hidden="hidden" i-amphtml-layout="nodisplay">
                <script type="application/json">[{{"selector":"#anim-1a95e072-cada-435a-afea-082ddd65ff10","keyframes":{{"opacity":[0,1]}},"delay":0,"duration":600,"easing":"cubic-bezier(0.2, 0.6, 0.0, 1)","fill":"both"}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" class="i-amphtml-layout-nodisplay" hidden="hidden" i-amphtml-layout="nodisplay">
                <script type="application/json">[{{"selector":"#anim-a938fe3f-03cf-47c5-9a84-da919c4f870b","keyframes":{{"transform":["translate3d(-115.2381%, 0px, 0)","translate3d(0px, 0px, 0)"]}},"delay":0,"duration":600,"easing":"cubic-bezier(0.2, 0.6, 0.0, 1)","fill":"both"}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" class="i-amphtml-layout-nodisplay" hidden="hidden" i-amphtml-layout="nodisplay">
                <script type="application/json">[{{"selector":"#anim-f7c5981e-ac77-48d5-9b40-7a987a3e2ab0","keyframes":{{"opacity":[0,1]}},"delay":0,"duration":600,"easing":"cubic-bezier(0.2, 0.6, 0.0, 1)","fill":"both"}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" class="i-amphtml-layout-nodisplay" hidden="hidden" i-amphtml-layout="nodisplay">
                <script type="application/json">[{{"selector":"#anim-0c1e94dd-ab91-415c-9372-0aa2e7e61630","keyframes":{{"transform":["translate3d(-115.55555%, 0px, 0)","translate3d(0px, 0px, 0)"]}},"delay":0,"duration":600,"easing":"cubic-bezier(0.2, 0.6, 0.0, 1)","fill":"both"}}]</script>
            </amp-story-animation>
            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" class="grid-layer i-amphtml-layout-container" i-amphtml-layout="container" style="--aspect-ratio:412/618;">
                <div class="page-fullbleed-area"><div class="page-safe-area">
                    <div class="_6120891"><div class="_89d52dd mask" id="el-f00095ab-c147-4f19-9857-72ac678f953f">
                        <div class="_dc67a5c fill"></div></div></div></div></div>
            </amp-story-grid-layer>
            <amp-story-grid-layer template="fill" class="i-amphtml-layout-container" i-amphtml-layout="container">
                <amp-video autoplay="autoplay" layout="fixed" width="1" height="1" poster="" id="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" cache="google" class="i-amphtml-layout-fixed i-amphtml-layout-size-defined" style="width:1px;height:1px" i-amphtml-layout="fixed">
                    <source type="audio/mpeg" src="{audio_url}">
                </amp-video>
            </amp-story-grid-layer>
            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" class="grid-layer i-amphtml-layout-container" i-amphtml-layout="container" style="--aspect-ratio:412/618;">
                <div class="page-fullbleed-area"><div class="page-safe-area">
                    <div class="_c19e533"><div class="_89d52dd mask" id="el-344ed989-789b-4a01-a124-9ae1d15d67f4">
                        <div data-leaf-element="true" class="_8aed44c">
                            <amp-img layout="fill" src="https://media.suvichaar.org/upload/polaris/polarisslide.png" alt="polarisslide.png" disable-inline-width="true" class="i-amphtml-layout-fill i-amphtml-layout-size-defined" i-amphtml-layout="fill"></amp-img>
                        </div></div></div>
                    <div class="_3d0c7a9"><div id="anim-1a95e072-cada-435a-afea-082ddd65ff10" class="_75da10d animation-wrapper">
                        <div id="anim-a938fe3f-03cf-47c5-9a84-da919c4f870b" class="_e559378 animation-wrapper">
                            <div id="el-2f080472-6c81-40a1-ac00-339cc8981388" class="_5342a26">
                                <h3 class="_d1a8d0d fill text-wrapper"><span><span class="_14af73e">{paragraph}</span></span></h3>
                            </div></div></div></div>
                    <div class="_a336742"><div id="anim-f7c5981e-ac77-48d5-9b40-7a987a3e2ab0" class="_75da10d animation-wrapper">
                        <div id="anim-0c1e94dd-ab91-415c-9372-0aa2e7e61630" class="_09239f8 animation-wrapper">
                            <div id="el-1a0d583c-c99b-4156-825b-3188408c0551" class="_ee8f788">
                                <h2 class="_59f9bb8 fill text-wrapper"><span><span class="_14af73e"></span></span></h2>
                            </div></div></div></div></div></div>
            </amp-story-grid-layer>
        </amp-story-page>
        """
    
    uploaded_html_file = st.file_uploader("📄 Upload AMP Template HTML (with <!--INSERT_SLIDES_HERE-->)", type=["html"], key="html_upload_tab3")
    uploaded_json_file = st.file_uploader("📦 Upload Output JSON", type=["json"], key="json_upload_tab3")

    if uploaded_html_file and uploaded_json_file:
        try:
            template_html = uploaded_html_file.read().decode("utf-8")
            output_data = json.load(uploaded_json_file)

            if "<!--INSERT_SLIDES_HERE-->" not in template_html:
                st.error("❌ Placeholder <!--INSERT_SLIDES_HERE--> not found in uploaded HTML.")
            else:
                all_slides = ""
                for key in sorted(output_data.keys(), key=lambda x: int(x.replace("slide", ""))):
                    slide_num = key.replace("slide", "")
                    data = output_data[key]
                    para_key = f"s{slide_num}paragraph1"
                    audio_key = f"audio_url{slide_num}"

                    if para_key in data and audio_key in data:
                        paragraph = data[para_key].replace("’", "'").replace('"', '&quot;')
                        audio_url = data[audio_key]
                        all_slides += generate_slide(paragraph, audio_url)

                final_html = template_html.replace("<!--INSERT_SLIDES_HERE-->", all_slides)
                filename = f"pre-final_amp_story_{int(time.time())}.html"

                st.success("✅ Final AMP HTML generated successfully!")
                st.download_button(
                    label="📥 Download Final AMP HTML",
                    data=final_html,
                    file_name=filename,
                    mime="text/html"
                )

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

with tab5:
    st.header("Content Submission Form")

    if "last_title" not in st.session_state:
        st.session_state.last_title = ""
        st.session_state.meta_description = ""
        st.session_state.meta_keywords = ""

    story_title = st.text_input("Story Title")
    
    if story_title.strip() and story_title != st.session_state.last_title:

        with st.spinner("Generating meta description, keywords, and filter tags..."):

            messages = [
                {
                    "role": "user",
                    "content": f"""
                    Generate the following for a web story titled '{story_title}':
                    1. A short SEO-friendly meta description
                    2. Meta keywords (comma separated)
                    3. Relevant filter tags (comma separated, suitable for categorization and content filtering)"""
                }
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.5,
                )
                output = response.choices[0].message.content
    
                # Extract metadata using regex
                desc = re.search(r"[Dd]escription\s*[:\-]\s*(.+)", output)
                keys = re.search(r"[Kk]eywords\s*[:\-]\s*(.+)", output)
                tags = re.search(r"[Ff]ilter\s*[Tt]ags\s*[:\-]\s*(.+)", output)
    
                st.session_state.meta_description = desc.group(1).strip() if desc else ""
                st.session_state.meta_keywords = keys.group(1).strip() if keys else ""
                st.session_state.generated_filter_tags = tags.group(1).strip() if tags else ""
    
            except Exception as e:
                st.warning(f"Error: {e}")
            st.session_state.last_title = story_title

    meta_description = st.text_area("Meta Description", value=st.session_state.meta_description)
    meta_keywords = st.text_input("Meta Keywords (comma separated)", value=st.session_state.meta_keywords)
    content_type = st.selectbox("Select your contenttype", ["News", "Article"])
    language = st.selectbox("Select your Language", ["en-US", "hi"])
    image_url = st.text_input("Enter your Image URL")
    uploaded_prefinal = st.file_uploader("💾 Upload pre‑final AMP HTML (optional)", type=["html","htm"], key="prefinal_upload")
    
    if uploaded_prefinal is None:
        st.error("Please upload a pre-final AMP HTML file before submitting.")

    categories = st.selectbox("Select your Categories", ["Art", "Travel", "Entertainment", "Literature", "Books", "Sports", "History", "Culture", "Wildlife", "Spiritual", "Food"])
    # Input field
    default_tags = [
        "Lata Mangeshkar",
        "Indian Music Legends",
        "Playback Singing",
        "Bollywood Golden Era",
        "Indian Cinema",
        "Musical Icons",
        "Voice of India",
        "Bharat Ratna",
        "Indian Classical Music",
        "Hindi Film Songs",
        "Legendary Singers",
        "Cultural Heritage",
        "Suvichaar Stories"
    ]

    tag_input = st.text_input(
        "Enter Filter Tags (comma separated):",
        value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)),
        help="Example: Music, Culture, Lata Mangeshkar"
    )

    use_custom_cover = st.radio("Do you want to add a custom cover image URL?", ("No", "Yes"))
    if use_custom_cover == "Yes":
        cover_image_url = st.text_input("Enter your custom Cover Image URL")
    else:
        cover_image_url = image_url  # fallback to image_url


    with st.form("content_form"):
        submit_button = st.form_submit_button("Submit")   # ← inside form

if submit_button:
    # Validation before processing
    missing_fields = []

    if not story_title.strip():
        missing_fields.append("Story Title")
    if not meta_description.strip():
        missing_fields.append("Meta Description")
    if not meta_keywords.strip():
        missing_fields.append("Meta Keywords")
    if not content_type.strip():
        missing_fields.append("Content Type")
    if not language.strip():
        missing_fields.append("Language")
    if not image_url.strip():
        missing_fields.append("Image URL")
    if not tag_input.strip():
        missing_fields.append("Filter Tags")
    if not categories.strip():
        missing_fields.append("Category")
    if not uploaded_prefinal:
        missing_fields.append("Raw HTML File")

    if missing_fields:
        st.error(f"❌ Please fill all required fields before submitting:\n- " + "\n- ".join(missing_fields))
    else:
        # ✅ All fields are valid, proceed with your full processing logic
        st.markdown("### Submitted Data")
        st.write(f"**Story Title:** {story_title}")
        st.write(f"**Meta Description:** {meta_description}")
        st.write(f"**Meta Keywords:** {meta_keywords}")
        st.write(f"**Content Type:** {content_type}")
        st.write(f"**Language:** {language}")

    key_path = "media/default.png"
    uploaded_url = ""

    try:
        nano, slug_nano, canurl, canurl1 = generate_slug_and_urls(story_title)
        page_title = f"{story_title} | Suvichaar"
    except Exception as e:

        st.error(f"Error generating canonical URLs: {e}")
        nano = slug_nano = canurl = canurl1 = page_title = ""

    # Image URL handling
    if image_url:

        filename = os.path.basename(urlparse(image_url).path)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".gif"]:
            ext = ".jpg"

        if image_url.startswith("https://stories.suvichaar.org/"):

            uploaded_url = image_url
            key_path = "/".join(urlparse(image_url).path.split("/")[2:])

        else:

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                unique_filename = f"{uuid.uuid4().hex}{ext}"
                s3_key = f"{s3_prefix}{unique_filename}"
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=response.content,
                    ContentType=response.headers.get("Content-Type", "image/jpeg"),
                )
                uploaded_url = f"{cdn_base_url}{s3_key}"
                key_path = s3_key
                st.success("Image uploaded successfully!")

            except Exception as e:
                st.warning(f"Failed to fetch/upload image. Using fallback. Error: {e}")
                uploaded_url = ""
    else:
        st.info("No Image URL provided. Using default.")

    try:
        # use the uploaded HTML as the working template
        html_template = uploaded_prefinal.read().decode("utf-8")
    
        user_mapping = {
            "Mayank": "https://www.instagram.com/iamkrmayank?igsh=eW82NW1qbjh4OXY2&utm_source=qr",
            "Onip": "https://www.instagram.com/onip.mathur/profilecard/?igsh=MW5zMm5qMXhybGNmdA==",
            "Naman": "https://njnaman.in/"
        }

        filter_tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
        category_mapping = {
            "Art": 1,
            "Travel": 2,
            "Entertainment": 3,
            "Literature": 4,
            "Books": 5,
            "Sports": 6,
            "History": 7,
            "Culture": 8,
            "Wildlife": 9,
            "Spiritual": 10
        }

        filternumber = category_mapping[categories]
        selected_user = random.choice(list(user_mapping.keys()))
        html_template = html_template.replace("{{user}}", selected_user)
        html_template = html_template.replace("{{userprofileurl}}", user_mapping[selected_user])
        html_template = html_template.replace("{{publishedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
        html_template = html_template.replace("{{modifiedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
        html_template = html_template.replace("{{storytitle}}", story_title)
        html_template = html_template.replace("{{metadescription}}", meta_description)
        html_template = html_template.replace("{{metakeywords}}", meta_keywords)
        html_template = html_template.replace("{{contenttype}}", content_type)
        html_template = html_template.replace("{{lang}}", language)
        html_template = html_template.replace("{{pagetitle}}", page_title)
        html_template = html_template.replace("{{canurl}}", canurl)
        html_template = html_template.replace("{{canurl1}}", canurl1)

        if image_url.startswith("http://media.suvichaar.org") or image_url.startswith("https://media.suvichaar.org"):
        
            html_template = html_template.replace("{{image0}}", image_url)

            parsed_cdn_url = urlparse(image_url)
            cdn_key_path = parsed_cdn_url.path.lstrip("/")  # ✅ Fix

            resize_presets = {
                "potraitcoverurl": (640, 853),
                "msthumbnailcoverurl": (300, 300),
            }

            for label, (width, height) in resize_presets.items():
                template = {
                    "bucket": bucket_name,
                    "key": cdn_key_path,
                    "edits": {
                        "resize": {
                            "width": width,
                            "height": height,
                            "fit": "cover"
                        }
                    }
                }
                encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                final_url = f"{cdn_prefix_media}{encoded}"
                # st.write(f"✅ Replacing {{{label}}} with {final_url}")
                html_template = html_template.replace(f"{{{label}}}", final_url)

        # Cleanup step to remove incorrect {url} wrapping
        html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\1"', html_template)
        html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\1"', html_template)

        st.markdown("### Final Modified HTML")
        st.code(html_template, language="html")

        # ----------- Generate and Provide Metadata JSON -------------
        metadata_dict = {
            "story_title": story_title,
            "categories": filternumber,
            "filterTags": filter_tags,
            "story_uid": nano,
            "story_link": canurl,
            "storyhtmlurl": canurl1,
            "urlslug": slug_nano,
            "cover_image_link": cover_image_url,
            "publisher_id": 1,
            "story_logo_link": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
            "keywords": meta_keywords,
            "metadescription": meta_description,
            "lang": language
        }

        s3_key = f"{slug_nano}.html"

        s3_client.put_object(
            Bucket="suvichaarstories",
            Key=s3_key,
            Body=html_template.encode("utf-8"),
            ContentType="text/html",
        )

        final_story_url = f"https://suvichaar.org/stories/{slug_nano}"  # This is your canurl
        st.success("✅ HTML uploaded successfully to S3!")
        st.markdown(f"🔗 **Live Story URL:** [Click to view your story]({final_story_url})")
        
        json_str = json.dumps(metadata_dict, indent=4)

        # Save data to session_state
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr(f"{slug_nano}.html", html_template)
            zip_file.writestr(f"{slug_nano}_metadata.json", json_str)
        
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download HTML + Metadata ZIP",
            data=zip_buffer,
            file_name=f"{story_title}.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error processing HTML: {e}")


with tab6:
        # ── AWS CONFIG ────────────────────────────────────────────────────
        AWS_ACCESS_KEY   = st.secrets["aws"]["AWS_ACCESS_KEY"]
        AWS_SECRET_KEY   = st.secrets["aws"]["AWS_SECRET_KEY"]
        AWS_REGION       = st.secrets["aws"]["AWS_REGION"]
        AWS_BUCKET       = st.secrets["aws"]["AWS_BUCKET"]
        S3_PREFIX        = st.secrets["aws"]["S3_PREFIX"]
        CDN_BASE         = st.secrets["aws"]["CDN_BASE"]
        CDN_PREFIX_MEDIA = "https://media.suvichaar.org/"
        
        # Initialize S3 client once
        s3 = boto3.client(
            "s3",
            aws_access_key_id     = AWS_ACCESS_KEY,
            aws_secret_access_key = AWS_SECRET_KEY,
            region_name           = AWS_REGION,
        )
        
        st.title("Cover Image Request")
        
        uploaded = st.file_uploader("📥 Upload Suvichaar JSON", type=["json"])
        if not uploaded:
            st.info("Please upload a Suvichaar‑style JSON to begin.")
            st.stop()
        
        # Parse & transform
        try:
            data = json.load(uploaded)
            transformed = {}
            for slide_key, info in data.items():
                idx = int(slide_key.replace("slide", ""))
                if "storytitle" in info:
                    text = info["storytitle"]
                elif "hookline" in info:
                    text = info["hookline"]
                else:
                    text = next((v for k, v in info.items() if "paragraph" in k), "")
                audio = info.get("audio_url", "")
        
                transformed[slide_key] = {
                    f"s{idx}paragraph1": text,
                    f"s{idx}audio1":    audio,
                    f"s{idx}image1":    "https://media.suvichaar.org/upload/polaris/polariscover.png",
                    f"s{idx}paragraph2":"Suvichaar"
                }
        
            st.success("✅ Transformation Complete")
            st.json(transformed)
        
        except json.JSONDecodeError:
            st.error("❌ Uploaded file is not valid JSON.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error during transformation: {e}")
            st.stop()
        
        # Generate thumbnail
        if st.button("Generate Thumbnail"):
            with st.spinner("Generating…"):
                try:
                    resp = requests.post(
                        "https://remotion.suvichaar.org/api/generate-news-thumbnail",
                        json=transformed,
                        timeout=30
                    )
                    resp.raise_for_status()
                except requests.RequestException as err:
                    st.error(f"Thumbnail API error: {err}")
                    st.stop()
        
            img_bytes = resp.content
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"{S3_PREFIX}cover_{ts}.png"
        
            # Upload without ACL
            try:
                s3.put_object(
                    Bucket      = AWS_BUCKET,
                    Key         = key,
                    Body        = img_bytes,
                    ContentType = resp.headers.get("Content-Type", "image/png"),
                )
            except Exception as s3_err:
                st.error(f"S3 upload failed: {s3_err}")
                st.stop()
        
            cdn_url = f"{CDN_PREFIX_MEDIA}{key}"
            st.success("🖼️ Thumbnail generated and uploaded!")
            st.markdown(f"[View on CDN]({cdn_url})")
            st.image(cdn_url, use_column_width=True)
        
            # Offer JSON download
            st.download_button(
                label="⬇️ Download Transformed JSON",
                data=json.dumps(transformed, indent=2, ensure_ascii=False),
                file_name=f"CoverJSON_{ts}.json",
                mime="application/json"
            )
        

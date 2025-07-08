import json
import time
import os
import uuid
import requests
import boto3
import nltk
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
    article = newspaper.Article(url)
    article.download()
    article.parse()

    try:
        article.nlp()
    except:
        pass

    # Fallbacks for missing fields
    title = article.title if article.title else "Untitled Article"
    text = article.text if article.text else "No article content available."
    summary = article.summary if article.summary else text[:300]

    # Final strip to ensure clean outputs
    return title.strip(), summary.strip(), text.strip()


def get_sentiment(text):
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def detect_category_and_subcategory(text):
    prompt = f"""
You are an expert news analyst.

Analyze the following news article and return:

1. category
2. subcategory
3. emotion

Article:
\"\"\"{text[:3000]}\"\"\"

Return as JSON:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Classify article into category, subcategory, and emotion."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.strip("```json").strip("```").strip()

    try:
        return json.loads(content)
    except:
        return {
            "category": "Unknown",
            "subcategory": "General",
            "emotion": "Neutral"
        }

def title_script_generator(category, subcategory, emotion, article_text, content_language="English", character_sketch=None):
    if not character_sketch:
        character_sketch = f"Polaris is a sincere and articulate {content_language} news anchor..."

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
{
  "slides": [
    { "title": "...", "prompt": "..." },
    ...
  ]
}
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

    if not article_text:
        article_text = "This article content could not be extracted properly."
    
    headline = article_text.split("\n")[0].strip().replace('"', '')

    if content_language == "Hindi":
    # Ask GPT to transliterate into Devanagari
        prompt = f"Transliterate the following Hindi sentence written in Latin script into Hindi Devanagari script:\n\nNamaskar doston, main hoon Polaris. Aaj ki badi khabar: {headline}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Hindi transliteration expert."},
                {"role": "user", "content": prompt}
            ]
        )
        slide1_script = response.choices[0].message.content.strip()
    else:
        slide1_script = f"Hello friends, I’m Polaris. Today’s big headline: {headline}"


    slides = [{
        "title": headline[:80],
        "prompt": "Intro slide with greeting and headline.",
        "image_prompt": f"Vector-style illustration of Polaris presenting news: {headline}",
        "script": slide1_script
    }]

    for slide in slides_raw:
        script_language = f"{content_language} (use Devanagari script)" if content_language == "Hindi" else content_language
        narration_prompt = f"""
            Write a narration in **{script_language}** (max 200 characters, including spaces/punctuation),
            in the voice of Polaris.
        
            Instruction: {slide['prompt']}
            Tone: Warm, simple, clear. No self-intro.
        
            Character sketch:
            {character_sketch}
        """
        
        narration_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write news narration in Hindi-English mix."},
                {"role": "user", "content": narration_prompt.strip()}
            ]
        )
        narration = narration_response.choices[0].message.content.strip()
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

# Tab 4 layout
def generate_hookline(title, summary):
    prompt = f"""
                You are a social media strategist. Your job is to create a short, attention-grabbing *hookline* for a news story.
                
                Title: {title}
                Summary: {summary}
                
                Requirements:
                - One sentence only
                - Avoid hashtags or emojis
                - Use simple and emotionally engaging language
                
                Example formats:
                - "What happened next will shock you."
                - "India's bold step in space tech."
                
                Now generate the hookline:
                """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You create viral hooklines for news stories."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    return response.choices[0].message.content.strip().replace('"', '')


def restructure_slide_output(final_output):
    slides = final_output.get("slides", [])
    structured = {}
    for idx, slide in enumerate(slides):
        key = f"s{idx + 1}paragraph1"
        structured[key] = slide.get("script", "").strip()
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

# === Streamlit UI ===
st.title("🧠 Web Story Content Generator")

tab1, tab2, tab3 ,tab4 ,tab5 ,tab6  = st.tabs(["📝 Slide Prompt Generator", "🔊 TTS Audio Generator", "Stoytitle/Hookline Insertor","🎞️ AMP Generator","Generate JSON", "Storyboard" ])

with tab1:
    st.title("🧠 Generalized Web Story Prompt Generator")
    url = st.text_input("Enter a news article URL")
    persona = st.selectbox(
        "Choose audience persona:",
        ["genz", "millenial", "working professionals", "creative thinkers", "spiritual explorers"]
    )
    content_language = st.selectbox("Choose content language", ["English", "Hindi"])
    number = st.number_input("Enter no of slides except (Storytitle and Hookline)", min_value=0, max_value=1000, value=10, step=1)

    if st.button("🚀 Submit and Generate JSON"):
        if url and persona:
            with st.spinner("Analyzing the article and generating prompts..."):
                try:
                    title, summary, full_text = extract_article(url)
                    sentiment = get_sentiment(summary or full_text)
                    result = detect_category_and_subcategory(full_text)
                    category, subcategory, emotion = result["category"], result["subcategory"], result["emotion"]
                    
                    hookline = generate_hookline(title, summary)
                    output = title_script_generator(category, subcategory, emotion, full_text, content_language)

                    final_output = {
                        "title": title,
                        "summary": summary,
                        "sentiment": sentiment,
                        "emotion": emotion,
                        "category": category,
                        "subcategory": subcategory,
                        "persona": persona,
                        "slides": output.get("slides", []),
                        "storytitle": title.strip(),
                        "hookline": hookline
                    }

                    structured_output = OrderedDict()
                    structured_output["storytitle"] = title.strip()

                    # Add s1paragraph1 to sXparagraph1
                    for i in range(1, number + 1):
                        key = f"s{i}paragraph1"
                        structured_output[key] = restructure_slide_output(final_output).get(key, "")

                    structured_output["hookline"] = hookline

                    # 🔁 If Hindi is selected → Transliterate using GPT
                    if content_language == "Hindi":
                        def transliterate_to_devanagari(json_data):
                            updated = {}
                            for k, v in json_data.items():
                                if k.startswith("s") and "paragraph1" in k:
                                    prompt = f"""Transliterate this Hindi sentence (written in Latin script) into Hindi Devanagari script. Return only the transliterated text:\n\n{v}"""
                                    response = client.chat.completions.create(
                                        model="gpt-4",
                                        messages=[
                                            {"role": "system", "content": "You are a Hindi transliteration expert."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    updated[k] = response.choices[0].message.content.strip()
                                else:
                                    updated[k] = v
                            return updated

                        structured_output = transliterate_to_devanagari(structured_output)

                    # ✅ Save & Download
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

    st.title("🧩 Tab 4: Inject Hookline & Title + Trim JSON")

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
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr("updated_test.html", updated_html)
                    zipf.writestr("output.json", json.dumps(updated_json, indent=2, ensure_ascii=False))
                buffer.seek(0)

                st.download_button(
                    label="⬇️ Download ZIP with HTML + JSON",
                    data=buffer,
                    file_name="tab4_output_bundle.zip",
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
    st.header("📥 Tab 5: Upload Suvichaar JSON → Renumber & Append Hookline")

    uploaded_file = st.file_uploader("Upload Suvichaar JSON file", type="json")

    if uploaded_file is not None:
        data = json.load(uploaded_file)

        # ── grab hookline & its audio ────────────────────────────────────────────
        hookline       = data.get("slide2", {}).get("hookline", "")
        hookline_audio = data.get("slide2", {}).get("audio_url", "")

        # ── collect paragraph+audio slides from slide3, slide4 … until break ────
        paragraph_slides = []
        new_idx = 1
        for i in range(3, 100):                          # covers slide3-slide99
            old_key = f"slide{i}"
            if old_key not in data:
                break                                    # stop at first missing
            slide_dict = data[old_key]

            # find the “sxparagraph1” key
            p_key = next((k for k in slide_dict if k.startswith("s") and "paragraph1" in k), None)
            if p_key and "audio_url" in slide_dict:
                paragraph_slides.append(
                    (
                        f"slide{new_idx}",
                        {
                            f"s{new_idx}paragraph1": slide_dict[p_key],
                            f"s{new_idx}audio1":     slide_dict["audio_url"],
                            f"s{new_idx}image1":    "https://media.suvichaar.org/upload/polaris/polariscover.png",
                            f"s{new_idx}paragraph2":"Suvichaar"
                        }
                    )
                )
                new_idx += 1

        # how many paragraph-audio blocks?
        st.success(f"✅ Found {len(paragraph_slides)} paragraph-audio blocks")

        # ── append the hookline as the final slide ──────────────────────────────
        hook_idx = new_idx
        hookline_slide = {
            f"s{hook_idx}paragraph1": hookline,
            f"s{hook_idx}audio1":     hookline_audio,
            f"s{hook_idx}image1":    "https://media.suvichaar.org/upload/polaris/polariscover.png",
            f"s{hook_idx}paragraph2":"Suvichaar"
        }

        # ── build the output JSON ───────────────────────────────────────────────
        output_data = dict(paragraph_slides)
        output_data[f"slide{hook_idx}"] = hookline_slide

        # ── preview + download ──────────────────────────────────────────────────
        st.subheader("✅ Final JSON Preview")
        st.json(output_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"remotion_{timestamp}.json"
        st.download_button(
            "📤 Download Final JSON",
            data=json.dumps(output_data, indent=4),
            file_name=file_name,
            mime="application/json"
        )

with tab6:
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


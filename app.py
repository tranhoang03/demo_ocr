import streamlit as st
import os
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

# Tắt log
os.environ["GGM_LOG_LEVEL"] = "error"

# Load model và handler từ HuggingFace (dùng cache để không tải lại mỗi lần)
@st.cache_resource
def load_model():
    chat_handler = MoondreamChatHandler.from_pretrained(
        repo_id="tranhoang03/Vintern-fine-tune-ss3",
        filename="*mmproj*.gguf"
    )
    llm = Llama.from_pretrained(
        repo_id="tranhoang03/Vintern-fine-tune-ss3",
        filename="Vintern-fine-tune-ss3-Q8_0.gguf",
        chat_handler=chat_handler,
        n_ctx=4096,
        log_level="ERROR",
        verbose=False
    )
    return llm

llm = load_model()

# Chuyển ảnh sang data URI để đưa vào prompt
def image_to_data_uri(image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Hàm chính xử lý trích xuất
def extract_info_from_image(image_bytes, user_prompt):
    if not image_bytes:
        return "Vui lòng upload ảnh hóa đơn!"
    
    uri = image_to_data_uri(image_bytes)
    prompt_text = user_prompt.strip() or "Hãy trích xuất từng thông tin trên hóa đơn này dưới dạng: 'Key': Value"
    
    messages = [
        {"role": "system", "content": "Bạn là trợ lý đọc và trích xuất thông tin từ hình ảnh."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": uri}}
        ]}
    ]
    try:
        resp = llm.create_chat_completion(messages=messages)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"

# Giao diện Streamlit
st.set_page_config(page_title="Trích xuất hóa đơn", layout="centered")

st.markdown("<h1 style='text-align:center; color:#00695c;'>📄 Trích xuất thông tin hóa đơn</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#004d40;'>Tải ảnh hóa đơn và nhận thông tin</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Tải ảnh hóa đơn", type=["jpg", "jpeg", "png"])

prompt = st.text_area(
    "Nhập prompt (tùy chọn)",
    placeholder="Ví dụ: 'Lấy ra số tiền thanh toán và mã hóa đơn'",
    height=80
)

if st.button("🚀 Trích xuất"):
    if uploaded_file:
        image_bytes = uploaded_file.read()
        with st.spinner("Đang xử lý..."):
            result = extract_info_from_image(image_bytes, prompt)
        st.subheader("📤 Kết quả trích xuất")
        st.code(result, language="json")
    else:
        st.warning("Vui lòng upload ảnh trước khi trích xuất.")

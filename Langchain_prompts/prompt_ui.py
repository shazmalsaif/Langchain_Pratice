from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

st.header("Reasearch Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


if st.button("Generate response"):
    if paper_input and style_input and length_input and paper_input != "Select...":
        with st.spinner("Generating response..."):
         prompt = (
            f"explain the research paper '{paper_input}'."
            f"The explanation style must be '{style_input}'"
            f"and the explanation length should be '{length_input}'."
         )
         result = model.invoke(prompt)
        st.write(result.content)

    else:
        st.warning("Please enter a topic to generate a response.")

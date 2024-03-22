import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from huggingface_hub import notebook_login


def getLLamaresponse(input_text,no_words,blog_style):

   
    llm=CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML',
                      model_type='llama',
                      config={'max_new_tokens':1000,
                              'temperature':0.01})
    

    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Blog Generator")

input_text=st.text_input("Enter the Blog Topic")


col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Coder','Researchers','Data Scientist','Common People','Software Developer','Writer','philosopher'),index=0)
    
submit=st.button("Generate")


if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
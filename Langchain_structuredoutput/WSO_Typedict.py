from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm )
#schema 
class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all key themes in the review."]
    smmary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[str,"The overall sentiment of the review, either positive or  negative."]
    pros:Annotated[list[str],"List of positive aspects mentioned in the review."]
    cons:Annotated[list[str],"List of negative aspects mentioned in the review."]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful

- Bulky and heavy—not great for one-handed use
- Bloatware still exists in One UI
- Expensive compared to competitors
""")

print(result)

### here The code doesn't work because the link chain is asking schema output For Open AI or like those type of models but this typing library doesn't work With hugging face deep sick model work with this deep sick model we have to use another library called pydantic which is used for data validation and settings management using python type annotations. It provides a way to define data models with type hints and validates the data against those models.
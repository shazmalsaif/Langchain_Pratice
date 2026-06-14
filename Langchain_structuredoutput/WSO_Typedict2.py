#from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
#from dotenv import load_dotenv
from pydantic import BaseModel,Field,EmailStr
from typing import Optional


class student(BaseModel):
    name:str = 'shazamal'
    age: Optional[str] = Field(default=18, description="BY DEFAULT AGE IS 18")
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=4.0, description="CGPA must be in range")



new_student = {'name':"saif", 'email':"Shazamal@example.com", 'cgpa':3.5}
student1 = student.model_validate(new_student) 


student_dict =dict(student1)
print(student1)
print(student_dict['name'])
 
student_json = student1.model_dump_json()
print(student_json)
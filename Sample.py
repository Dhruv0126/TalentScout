
import streamlit as st
from dotenv import load_dotenv
import os
import json
import re
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence


load_dotenv()
# Access the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Debugging: Verify the key is loaded
print("Groq API Key:", groq_api_key)


class TalentScoutGroq:
    def __init__(self):
        self.skill_categories = {
            'Software Development': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 'TypeScript', 'Swift'],
            'Web Development': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot', 'ASP.NET Core'],
            'Cloud & DevOps': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Terraform'],
            'Data Science & AI': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn', 'Keras'],
            'Mobile Development': ['Android', 'iOS', 'React Native', 'Flutter', 'Kotlin', 'Swift']
        }

        self.llm = ChatGroq(
            api_key="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",
            model_name="llama-3.1-8b-instant",
            temperature=0.7
        )

        self.question_prompt = PromptTemplate(
            input_variables=['role', 'skills', 'experience'],
            template=(
                "Generate exactly five interview questions for a {role} with {experience} years of experience.\n"
                "Skills: {skills}\n\n"
                "Output the result as a JSON array of objects. Each object must have two keys: "
                "'id' (an integer starting at 1) and 'question' (the text of the question).\n"
                "Ensure that the output is valid JSON and nothing else."
            )
        )

    def generate_ai_interview_questions(self, candidate):
        skills_str = ', '.join(candidate.get('skills', []))
        chain = self.question_prompt | self.llm

        inputs = {
            "role": candidate.get('role', ''),
            "skills": skills_str,
            "experience": str(candidate.get('experience', 0))
        }
        
        try:
            response = chain.invoke(inputs)
            response_text = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_text
            questions = json.loads(json_str)
            return questions[:5] if isinstance(questions, list) else []
        except json.JSONDecodeError:
            return []

    def save_interview_results(self, candidate, questions, conversation):
        results = {"candidate": candidate, "questions": questions, "conversation": conversation}
        filename = f"interview_results_{candidate.get('name', 'candidate')}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        st.success(f"Interview results saved to {filename}")

    def run_streamlit_app(self):
        st.title("TalentScout AI Interview Chat")
        candidate = {
            'name': st.text_input("Candidate Name"),
            'email': st.text_input("Email Address"),
            'phone': st.text_input("Phone Number"),
            'experience': st.number_input("Years of Experience", min_value=0, max_value=50, value=0),
            'role': st.selectbox("Select Target Role", ['Software Engineer', 'Data Scientist', 'DevOps Engineer']),
            'skills': []
        }
        
        st.header("Select Skills")
        for category, skills in self.skill_categories.items():
            st.subheader(category)
            selected = st.multiselect(f"Select {category} Skills", skills)
            candidate['skills'].extend(selected)
        
        if st.button("Start Interview Chat"):
            if not candidate['name'] or not candidate['role'] or not candidate['skills']:
                st.warning("Please fill in all required fields (Name, Role, and Skills).")
                return
            st.session_state["questions"] = self.generate_ai_interview_questions(candidate)
            st.session_state["conversation"] = []
            st.session_state["current_question"] = 0
            st.session_state["candidate"] = candidate
        
        if "questions" in st.session_state:
            st.header("Interview Chat")
            conv = st.session_state["conversation"]
            questions = st.session_state["questions"]
            current_index = st.session_state["current_question"]

            for msg in conv:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            if current_index < len(questions):
                current_question = questions[current_index]
                with st.chat_message("assistant"):
                    st.write(f"Q{current_question['id']}: {current_question['question']}")
                
                answer = st.chat_input("Type your answer here...")
                if answer:
                    st.session_state["conversation"].append({"role": "user", "content": answer})
                    st.session_state["conversation"].append({"role": "assistant", "content": "Answer recorded."})
                    st.session_state["current_question"] += 1
                    
                    if st.session_state["current_question"] >= len(questions):
                        st.success("You have answered all questions!")
                        self.save_interview_results(
                            candidate=st.session_state.get("candidate", {}),
                            questions=st.session_state.get("questions", []),
                            conversation=st.session_state.get("conversation", [])
                        )
            

def main():
    talent_scout = TalentScoutGroq()
    talent_scout.run_streamlit_app()

if __name__ == "__main__":
    main()

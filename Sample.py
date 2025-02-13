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

# Custom CSS for tech-themed design
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

class TalentScoutGroq:
    def __init__(self):
        self.skill_categories = {
            'Software Development': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 'TypeScript', 'Swift'],
            'Web Development': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot', 'ASP.NET Core'],
            'Cloud & DevOps': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Terraform'],
            'Data Science & AI': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn', 'Keras'],
            'Mobile Development': ['Android', 'iOS', 'React Native', 'Flutter', 'Kotlin', 'Swift']
        }

        # Updated prompt template using 'desired_position'
        self.question_prompt = PromptTemplate(
            input_variables=['desired_position', 'skills', 'experience'],
            template=(
                "Generate exactly five interview questions for a candidate applying for {desired_position} "
                "with {experience} years of experience.\n"
                "Skills: {skills}\n\n"
                "Output the result as a JSON array of objects. Each object must have two keys: "
                "'id' (an integer starting at 1) and 'question' (the text of the question).\n"
                "Ensure that the output is valid JSON and nothing else."
            )
        )

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.7
        )

    def generate_ai_interview_questions(self, candidate):
        skills_str = ', '.join(candidate.get('skills', []))
        chain = self.question_prompt | self.llm

        inputs = {
            "desired_position": candidate.get('desired_position', ''),
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
        
        # Sidebar with instructions
        st.sidebar.title("Instructions")
        st.sidebar.markdown(
            "Fill in your details, select your skills, and click **Start Interview Chat**. \n\n"
            "During the interview, if you wish to exit, simply type **exit**, **quit**, or **bye** in the chat input."
        )

        st.markdown("""
            <div style="background-color: #0E1117; padding: 10px; border-radius: 10px;">
                <h2 style="color: #00FF00;">Welcome to TalentScout AI</h2>
                <p style="color: #FFFFFF;">Your AI-powered interview assistant.</p>
            </div>
        """, unsafe_allow_html=True)

        # Candidate Information Form
        with st.container():
            st.header("Candidate Information")
            candidate = {
                'name': st.text_input("Full Name", placeholder="Enter your full name"),
                'email': st.text_input("Email Address", placeholder="Enter your email"),
                'phone': st.text_input("Phone Number", placeholder="Enter your phone number"),
                'experience': st.number_input("Years of Experience", min_value=0, max_value=50, value=0),
                'desired_position': st.text_input("Desired Position(s)", placeholder="Enter your desired position(s)"),
                'current_location': st.text_input("Current Location", placeholder="Enter your current location"),
                'skills': []
            }
        
        # Tech Stack Selection
        with st.container():
            st.header("Tech Stack Declaration")
            for category, skills in self.skill_categories.items():
                st.subheader(category)
                selected = st.multiselect(f"Select {category} Skills", skills)
                candidate['skills'].extend(selected)
        
        # Start Interview Button with validation and initial greeting
        if st.button("Start Interview Chat", key="start_interview"):
            if not candidate['name'] or not candidate['desired_position'] or not candidate['skills'] or not candidate['current_location']:
                st.warning("Please fill in all required fields: Full Name, Desired Position(s), Current Location, and at least one Skill.")
                return
            st.session_state["questions"] = self.generate_ai_interview_questions(candidate)
            # Initialize conversation with a greeting from the assistant
            st.session_state["conversation"] = [{
                "role": "assistant", 
                "content": "Hello, I'm TalentScout AI Interview Assistant. Welcome to your interview session. "
                           "If you wish to exit at any time, please type 'exit', 'quit', or 'bye'."
            }]
            st.session_state["current_question"] = 0
            st.session_state["candidate"] = candidate
        
        # Interview Chat Area
        if "questions" in st.session_state:
            st.header("Interview Chat")
            conv = st.session_state["conversation"]
            questions = st.session_state["questions"]
            current_index = st.session_state["current_question"]

            # Display conversation history
            for msg in conv:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Display the next question if available
            if current_index < len(questions):
                current_question = questions[current_index]
                with st.chat_message("assistant"):
                    st.write(f"Q{current_question['id']}: {current_question['question']}")
                
                answer = st.chat_input("Type your answer here...")
                if answer:
                    # Check for conversation-ending keywords
                    if answer.strip().lower() in ['exit', 'quit', 'bye']:
                        st.session_state["conversation"].append({"role": "user", "content": answer})
                        st.session_state["conversation"].append({"role": "assistant", "content": "Thank you for your time. The conversation has been ended. We appreciate your interest."})
                        self.save_interview_results(
                            candidate=st.session_state.get("candidate", {}),
                            questions=st.session_state.get("questions", []),
                            conversation=st.session_state.get("conversation", [])
                        )
                        st.stop()
                    else:
                        st.session_state["conversation"].append({"role": "user", "content": answer})
                        st.session_state["conversation"].append({"role": "assistant", "content": "Answer recorded."})
                        st.session_state["current_question"] += 1
                        
                        # If all questions have been answered, conclude the interview
                        if st.session_state["current_question"] >= len(questions):
                            st.success("You have answered all the questions!")
                            st.session_state["conversation"].append({"role": "assistant", "content": "Thank you for completing the interview. We will be in touch with you regarding the next steps."})
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
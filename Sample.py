import streamlit as st
from dotenv import load_dotenv
import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

load_dotenv()
# Access the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()


if groq_api_key:
    print("Groq API Key Loaded Successfully")
else:
    print("Groq API Key Missing")

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

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Resume evaluation prompt
        self.resume_eval_prompt = PromptTemplate(
            input_variables=['resume_text', 'detected_skills'],
            template=(
                "Analyze the following candidate resume and provide a detailed evaluation:\n\n"
                "Resume Text:\n{resume_text}\n\n"
                "Detected Skills: {detected_skills}\n\n"
                "Please provide:\n"
                "1. Candidate Summary\n"
                "2. Technical Skills Assessment\n"
                "3. Experience Level\n"
                "4. Project Quality\n"
                "5. Strengths\n"
                "6. Weaknesses\n"
                "7. Hiring Recommendation\n"
                "8. Suggested Interview Questions (3-5)\n\n"
                "Output in JSON format with keys: summary, skills_assessment, experience_level, project_quality, strengths, weaknesses, recommendation, interview_questions"
            )
        )

    def generate_ai_interview_questions(self, candidate):
        candidate['skills'] = list(set(candidate.get('skills', [])))
        skills_str = ', '.join(candidate['skills'])
        chain = self.question_prompt | self.llm

        inputs = {
            "desired_position": candidate.get('desired_position', ''),
            "skills": skills_str,
            "experience": str(candidate.get('experience', 0))
        }
        
        try:
            response = chain.invoke(inputs)
        
            response_text = (
                response.content
                if hasattr(response, 'content')
                else str(response)
            )
        
            json_match = re.search(
                r'\[.*?\]',
                response_text,
                re.DOTALL
            )
        
            json_str = (
                json_match.group(0)
                if json_match
                else response_text
            )
        
            questions = json.loads(json_str)
        
            return questions[:5] if isinstance(questions, list) else []
        
        except json.JSONDecodeError:
            return []
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            return []

    def save_interview_results(self, candidate, questions, conversation):
        results = {"candidate": candidate, "questions": questions, "conversation": conversation}
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', candidate.get('name', 'candidate'))
        filename = f"interview_results_{safe_name}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        st.success(f"Interview results saved to {filename}")

    def parse_resume(self, uploaded_file):
        """Parse PDF or DOCX resume and extract text."""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            raise ValueError("Unsupported file type")
        return text

    def preprocess_text(self, text):
        """Clean and preprocess the extracted text."""
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text

    def detect_skills(self, text):
        """Detect skills from the preprocessed text using keyword matching."""
        detected_skills = []
        all_skills = []
        for category, skills in self.skill_categories.items():
            all_skills.extend(skills)
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]
        
        for skill in all_skills:
            if skill.lower() in filtered_words:
                detected_skills.append(skill)
        
        return list(set(detected_skills))  # Remove duplicates

    def extract_candidate_details(self, text):
        """Extract candidate details from resume text using regex and simple patterns."""
        details = {
            'name': '',
            'email': '',
            'phone': '',
            'experience': 0,
            'desired_position': '',
            'current_location': '',
            'skills': []
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            details['email'] = emails[0]
        
        # Extract phone (simple pattern for US/International)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            details['phone'] = phones[0]
        
        # Extract experience (look for years)
        exp_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience'
        exps = re.findall(exp_pattern, text, re.IGNORECASE)
        if exps:
            details['experience'] = int(exps[0])
        
        # Extract name (assume first line or after contact)
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and not re.search(r'@|phone|email|address', line, re.IGNORECASE) and len(line.split()) <= 4:
                details['name'] = line
                break
        
        # Extract location (look for city, state)
        loc_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b'
        locs = re.findall(loc_pattern, text)
        if locs:
            details['current_location'] = ', '.join(locs[0])
        
        # Extract position (look for job titles)
        position_keywords = ['developer', 'engineer', 'analyst', 'scientist', 'manager', 'specialist', 'consultant']
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in position_keywords):
                details['desired_position'] = line.strip()
                break
        
        # Skills detection
        details['skills'] = self.detect_skills(text)
        
        return details

    def evaluate_resume_with_llm(self, resume_text, detected_skills):
        """Use LLM to evaluate the resume."""
        chain = self.resume_eval_prompt | self.llm
        inputs = {
            "resume_text": resume_text,
            "detected_skills": ', '.join(detected_skills)
        }
        try:
            response = chain.invoke(inputs)
            response_text = response.content if hasattr(response, 'content') else str(response)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else response_text
            evaluation = json.loads(json_str)
            return evaluation
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw_response": response_text}
        except Exception as e:
            return {"error": f"LLM evaluation failed: {e}", "raw_response": str(e)}

    def run_streamlit_app(self):
        st.markdown(
            '''
            <style>
            .gradient-title {
                font-size: 2.8rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 0.5em;
                background: linear-gradient(270deg, #7C4DFF, #8F00FF, #563C5C, #00FFEA, #FF61A6, #7C4DFF);
                background-size: 200% 200%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                color: transparent;
                text-shadow: 0 2px 8px rgba(44,0,60,0.18);
                animation: gradientMove 4s ease-in-out infinite;
            }
            @keyframes gradientMove {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            </style>
            <h1 class="gradient-title">TalentScout AI</h1>
            ''',
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(["AI Interview Chat", "Resume Analysis"])

        with tab1:
            self.run_interview_app()

        with tab2:
            self.run_resume_analysis_app()

    def run_interview_app(self):
        st.sidebar.title("Instructions")
        st.sidebar.markdown(
            "Upload your resume to auto-fill the form, or fill manually. Then select skills and start the interview. \n\n"
            "During the interview, answer questions one by one. Type **exit**, **quit**, or **bye** to end."
        )

        st.markdown(
            """
            <div style="background: linear-gradient(90deg, #563C5C 0%, #7C4DFF 100%); padding: 16px; border-radius: 12px; margin-bottom: 1.5em; text-align:center;">
                <h2 style="color: #FFFFFF; font-size: 2rem; font-weight: 600; margin-bottom: 0.2em;">Welcome to <span style='color:#7C4DFF;'>TalentScout AI</span></h2>
                <p style="color: #F3EFFF; font-size: 1.1rem;">Your AI-powered interview assistant.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Resume Upload for Auto-fill
        with st.container():
            st.header("Resume Upload (Optional)")
            uploaded_resume = st.file_uploader("Upload your resume (PDF/DOCX) to auto-fill the form", type=["pdf", "docx"], key="resume_upload")
            if uploaded_resume and st.button("Extract Details from Resume"):
                try:
                    raw_text = self.parse_resume(uploaded_resume)
                    extracted_details = self.extract_candidate_details(raw_text)
                    st.session_state.update(extracted_details)
                    st.success("Details extracted and form updated!")
                except Exception as e:
                    st.error(f"Error extracting details: {e}")

        # Candidate Information Form
        with st.container():
            st.header("Candidate Information")
            candidate = {
                'name': st.text_input("Full Name", value=st.session_state.get('name', ''), placeholder="Enter your full name"),
                'email': st.text_input("Email Address", value=st.session_state.get('email', ''), placeholder="Enter your email"),
                'phone': st.text_input("Phone Number", value=st.session_state.get('phone', ''), placeholder="Enter your phone number"),
                'experience': st.number_input("Years of Experience", value=st.session_state.get('experience', 0), min_value=0, max_value=50),
                'desired_position': st.text_input("Desired Position(s)", value=st.session_state.get('desired_position', ''), placeholder="Enter your desired position(s)"),
                'current_location': st.text_input("Current Location", value=st.session_state.get('current_location', ''), placeholder="Enter your current location"),
                'skills': st.session_state.get('skills', [])
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

            # If we need to ask the next question
            if current_index < len(questions):
                # If the last message is not the current question, append it
                if not (len(conv) > 0 and conv[-1]["role"] == "assistant" and conv[-1]["content"].startswith(f"Q{questions[current_index]['id']}:")):
                    conv.append({
                        "role": "assistant",
                        "content": f"Q{questions[current_index]['id']}: {questions[current_index]['question']}"
                    })
                    st.session_state["conversation"] = conv
                    st.rerun()

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
                        st.rerun()

    def run_resume_analysis_app(self):
        st.header("Resume Analysis")
        st.markdown("Upload a resume (PDF or DOCX) for AI-powered evaluation.")

        uploaded_file = st.file_uploader("Choose a resume file", type=["pdf", "docx"])

        if uploaded_file is not None:
            try:
                # Step 1 & 2: Parse resume
                raw_text = self.parse_resume(uploaded_file)
                details = self.extract_candidate_details(raw_text)

                st.subheader("Candidate Details")
                st.markdown(f"**Name:** {details.get('name', 'N/A')}")
                st.markdown(f"**Email:** {details.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {details.get('phone', 'N/A')}")
                st.markdown(f"**Experience:** {details.get('experience', 0)} years")
                st.markdown(f"**Desired Position:** {details.get('desired_position', 'N/A')}")
                st.markdown(f"**Current Location:** {details.get('current_location', 'N/A')}")
                st.markdown(f"**Detected Skills:** {', '.join(details.get('skills', [])) if details.get('skills') else 'No skills detected'}")

                # Step 3: Preprocess
                cleaned_text = self.preprocess_text(raw_text)

                # Step 4: Skill Detection
                detected_skills = self.detect_skills(cleaned_text)
                st.subheader("Skill Detection")
                st.write(", ".join(detected_skills) if detected_skills else "No skills detected")

                # Step 5-7: LLM Evaluation
                if st.button("Analyze Resume"):
                    with st.spinner("Analyzing resume with AI..."):
                        evaluation = self.evaluate_resume_with_llm(raw_text, detected_skills)

                    # Step 8: Display Results
                    if "error" in evaluation:
                        st.error(evaluation["error"])
                        st.text(evaluation["raw_response"])
                    else:
                        st.subheader("AI Evaluation")
                        st.write(f"**Summary:** {evaluation.get('summary', 'N/A')}")
                        st.write(f"**Skills Assessment:** {evaluation.get('skills_assessment', 'N/A')}")
                        st.write(f"**Experience Level:** {evaluation.get('experience_level', 'N/A')}")
                        st.write(f"**Project Quality:** {evaluation.get('project_quality', 'N/A')}")
                        st.write(f"**Strengths:** {evaluation.get('strengths', 'N/A')}")
                        st.write(f"**Weaknesses:** {evaluation.get('weaknesses', 'N/A')}")
                        st.write(f"**Recommendation:** {evaluation.get('recommendation', 'N/A')}")
                        st.write("**Suggested Interview Questions:**")
                        questions = evaluation.get('interview_questions', [])
                        if isinstance(questions, list):
                            for q in questions:
                                st.write(f"- {q}")
                        else:
                            st.write(questions)

            except Exception as e:
                st.error(f"Error processing resume: {e}")

def main():
    talent_scout = TalentScoutGroq()
    talent_scout.run_streamlit_app()

if __name__ == "__main__":
    main()


# TalentScout

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-blueviolet?logo=streamlit)](https://talentscout-dhruv0126.streamlit.app/)

Access the live deployed app here: [https://talentscout-dhruv0126.streamlit.app/](https://talentscout-dhruv6012.streamlit.app/)

## TalentScout AI - Interview & Resume Analysis Platform 🤖

An AI-powered recruitment platform featuring intelligent technical interviews and automated resume evaluation using Groq's Llama 3.1 model. Streamline candidate assessments with AI-driven insights and personalized question generation.

## 🚀 Key Features

### 📋 AI Interview Chat Tab
1. **Resume Auto-Fill** - Upload PDF/DOCX resume to auto-populate candidate form
2. **AI-Powered Question Generation** - Dynamic interview questions based on candidate skills and experience
3. **Tech Stack Declaration** - Multi-category skill selection across 5 technical domains
4. **Interactive Chat Interface** - Real-time interview simulation with conversation logging
5. **Session Recording** - Automatic JSON export of complete interview sessions
6. **Smart Exit System** - Natural language commands to end sessions (`exit/quit/bye`)

### 📄 Resume Analysis Tab
1. **Resume Parsing** - Extract text from PDF/DOCX formats
2. **Candidate Details Extraction** - Auto-detect name, email, phone, experience, location, position
3. **NLP Skill Detection** - Identify programming languages, frameworks, tools, and technologies
4. **Text Preprocessing** - Clean and normalize resume text for analysis
5. **AI Evaluation** - LLM-powered comprehensive resume assessment including:
   - Candidate summary
   - Technical skills assessment
   - Experience level evaluation
   - Project quality analysis
   - Strengths and weaknesses identification
   - Hiring recommendation
   - Suggested interview questions

## ⚙️ Installation

1. Clone repository:
```bash
git clone https://github.com/Dhruv0126/talentscout-ai.git
cd TalentScout
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (local development):
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

## 🎯 Usage

### Local Development
1. Start the application:
```bash
streamlit run Sample.py
```

2. Open browser at `http://localhost:8501`

### AI Interview Chat
- Upload resume (optional) to auto-fill candidate form
- Or manually fill candidate information
- Select technical skills from multiple categories
- Click `Start Interview Chat`
- Answer 5 AI-generated personalized questions
- Session auto-saves when complete or type `exit/quit/bye` to end

### Resume Analysis
- Upload a resume (PDF or DOCX)
- View extracted candidate details
- Check automatically detected skills
- Click `Analyze Resume` for AI evaluation
- Review comprehensive assessment report

## 🔧 Configuration

### Local Environment Variables
```env
GROQ_API_KEY=your_api_key_here
```

### Streamlit Cloud Deployment
Add `GROQ_API_KEY` in Streamlit Cloud secrets:
1. Go to app settings → Secrets
2. Add: `GROQ_API_KEY = "your_rotated_api_key"`
3. Deploy

### Customize Skill Categories
Modify `skill_categories` in [Sample.py](Sample.py) `TalentScoutGroq` class:
```python
self.skill_categories = {
    'Your Category': ['Skill1', 'Skill2', 'Skill3'],
    # ... existing categories
}
```

## 🛠️ Technologies Used

- **Core AI**: Groq LPU + Llama-3.1-8b-instant
- **Framework**: Streamlit 1.45.0+
- **LangChain**: Prompt templating & chain construction
- **Document Parsing**: PyPDF2, python-docx
- **NLP**: NLTK, spacy-compatible tokenization
- **Data Handling**: Pandas, JSON
- **Security**: Python-dotenv for environment management

## 📦 Dependencies

```
streamlit>=1.45.0
python-dotenv>=1.0.1
langchain>=0.3.25
langchain-core>=0.3.60
langchain-groq>=0.3.2
pydantic>=2.11.0
PyPDF2
python-docx
nltk
pandas
```

## 🔒 Security & Best Practices

- ✅ `.env` file is ignored from Git (see `.gitignore`)
- ✅ Never commit API keys to repository
- ✅ Rotate exposed keys immediately
- ✅ Use Streamlit Secrets for cloud deployment
- ✅ Interview results stored locally as JSON

## 📊 Workflow Overview

### Interview Chat Flow
1. User fills candidate info or uploads resume for auto-fill
2. Selects technical skills
3. AI generates 5 personalized questions
4. User answers each question
5. Session recorded and exported as JSON

### Resume Analysis Flow
1. User uploads PDF/DOCX resume
2. System extracts text and parses candidate details
3. NLP identifies technical skills
4. Text preprocessing normalizes content
5. Prompt engineering prepares evaluation request
6. LLM generates comprehensive assessment
7. Results displayed in dashboard

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Groq for their revolutionary LPU inference engine
- Streamlit for interactive web framework
- LangChain team for LLM integration patterns
- NLTK community for NLP tools

---

**Contact Developer**:  
Dhruv Gupta  
📧 [dhruv06012@gmail.com](mailto:dhruv06012@gmail.com)  
💼 [LinkedIn Profile](https://linkedin.com/in/dhruvgupta0126)  
🐙 [GitHub Profile](https://github.com/Dhruv0126)  
🔗 [Live Demo](https://talentscout-dhruv0126.streamlit.app/)



```markdown
## TalentScout AI Interview Assistant 🤖

An AI-powered interview platform that generates personalized technical questions using Groq's Llama 3.1 model, designed to streamline technical candidate assessments.

## 🚀 Key Features

1. AI-Powered Question Generation
  Dynamic interview questions based on candidate skills and experience
2. Tech Stack Declaration
  Multi-category skill selection across 5 technical domains
3. Interactive Chat Interface  
  Real-time interview simulation with conversation logging
4. Session Recording  
  Automatic JSON export of complete interview sessions
5. Smart Exit System
  Natural language commands to end sessions (`exit/quit/bye`)

## ⚙️ Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/talentscout-ai.git
cd talentscout-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

## 🎯 Usage

1. Start the application:
```bash
streamlit run Sample.py
```

2. In the browser:
- Fill candidate information
- Select technical skills
- Click `Start Interview Chat`

3. Interview flow:
- AI generates 5 personalized questions
- Answer questions in chat interface
- Session auto-saves when complete

## 🔧 Configuration

### Environment Variables
```env
GROQ_API_KEY=sk-your-api-key-here  # Get from Groq Cloud
```

### Customize Skill Categories
Modify `skill_categories` in `TalentScoutGroq` class:
```python
self.skill_categories = {
    'New Category': ['Skill1', 'Skill2'],
    # ... existing categories
}
```

## 🛠️ Technologies Used

- **Core AI**: Groq LPU + Llama-3.1-8b-instant
- **Framework**: Streamlit 🎈
- **LangChain**: Prompt templating & chain construction
- **Data Handling**: JSON interview logging
- **Security**: Dotenv configuration



## 🙏 Acknowledgments

- Groq for their revolutionary inference engine
- Streamlit for interactive web framework
- LangChain team for LLM integration patterns

---

**Contact Developer**:  
Dhruv Gupta  
📧 [dhruv06012@gmail.com](mailto:dhruv06012@gmail.com)  
💼 [LinkedIn Profile](https://linkedin.com/in/dhruvgupta0126)  
🐙 [GitHub Profile](https://github.com/Dhruv0126)


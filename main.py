import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from groq import Groq

# === Load API Key from .env ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

groq_llm = Groq(api_key=groq_api_key)

# === Define Agent State ===
class AgentState(TypedDict):
    topic: str
    trends: str
    competitor_analysis: str
    content_ideas: str
    summary: str

# === Node Functions with Persona Prompts ===
def trend_researcher_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    prompt = f"""You are a Trend Research Analyst with 10+ years of experience.
    Identify the latest and emerging digital content trends around the niche: {topic}.
    Be specific and list at least 3 current trend insights."""
    
    response = groq_llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    return {**state, "trends": response.choices[0].message.content.strip()}

def competitor_analyst_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    prompt = f"""You're a Competitor Analyst. Examine popular competitors in the '{topic}' niche.
    What types of content are they producing? What seems to be working well based on engagement and frequency?
    Provide 2-3 concrete examples."""
    
    response = groq_llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return {**state, "competitor_analysis": response.choices[0].message.content.strip()}

def idea_generator_node(state: AgentState) -> AgentState:
    prompt = f"""You're a Creative Director. Based on the topic '{state['topic']}', recent trends ({state['trends']}),
    and competitor analysis ({state['competitor_analysis']}), generate 5 unique, creative, viral-ready content ideas 
    for Instagram, LinkedIn, or YouTube. Label each idea clearly."""
    
    response = groq_llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return {**state, "content_ideas": response.choices[0].message.content.strip()}

def summary_agent_node(state: AgentState) -> AgentState:
    prompt = f"""You're a content strategist. Given the following data:
    - Topic: {state['topic']}
    - Trends: {state['trends']}
    - Competitor Analysis: {state['competitor_analysis']}
    - Content Ideas: {state['content_ideas']}
    
    Write a short 3-paragraph summary explaining the content strategy for a client unfamiliar with digital marketing."""
    
    response = groq_llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return {**state, "summary": response.choices[0].message.content.strip()}

# === Build LangGraph ===
graph = StateGraph(AgentState)
graph.add_node("TrendResearcher", RunnableLambda(trend_researcher_node))
graph.add_node("CompetitorAnalyst", RunnableLambda(competitor_analyst_node))
graph.add_node("IdeaGenerator", RunnableLambda(idea_generator_node))
graph.add_node("SummaryAgent", RunnableLambda(summary_agent_node))

graph.set_entry_point("TrendResearcher")
graph.add_edge("TrendResearcher", "CompetitorAnalyst")
graph.add_edge("CompetitorAnalyst", "IdeaGenerator")
graph.add_edge("IdeaGenerator", "SummaryAgent")
graph.add_edge("SummaryAgent", END)

content_graph = graph.compile()

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Run the content generation graph
        result = content_graph.invoke({"topic": topic})
        
        return jsonify({
            'success': True,
            'data': {
                'topic': topic,
                'trends': result.get('trends', ''),
                'competitor_analysis': result.get('competitor_analysis', ''),
                'content_ideas': result.get('content_ideas', ''),
                'summary': result.get('summary', '')
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
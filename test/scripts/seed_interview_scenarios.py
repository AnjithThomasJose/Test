import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.interview_chroma import interview_chroma

def seed_scenarios():
    """Seed default interview scenarios to ChromaDB"""
    
    scenarios = [
        # OPERATIONS_MANAGEMENT scenarios
        {
            "scenario_id": "ops_ps_001",
            "role_category": "OPERATIONS_MANAGEMENT",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Production alert: Your team's delivery metrics dropped 40% this week. Walk me through your investigation approach.",
            "expected_depth": ["root_cause", "data_analysis", "stakeholder_comm"],
            "follow_up_triggers": ["deeper_probe", "corrective"]
        },
        {
            "scenario_id": "ops_ps_002",
            "role_category": "OPERATIONS_MANAGEMENT",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "You need to implement a new process across 5 different teams. How do you ensure adoption and measure success?",
            "expected_depth": ["change_management", "metrics", "communication"],
            "follow_up_triggers": ["escalation"]
        },
        {
            "scenario_id": "ops_beh_001",
            "role_category": "OPERATIONS_MANAGEMENT",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Tell me about a time you had to deliver bad news to stakeholders. How did you handle it? (STAR format)",
            "expected_depth": ["situation", "task", "action", "result"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "ops_beh_002",
            "role_category": "OPERATIONS_MANAGEMENT",
            "stage": "behavioral",
            "difficulty": "intermediate",
            "question_text": "Describe a situation where you had to manage conflicting priorities from different departments. How did you resolve it?",
            "expected_depth": ["stakeholder_management", "prioritization", "communication"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        
        # TECHNICAL_CODING scenarios
        {
            "scenario_id": "tech_ps_001",
            "role_category": "TECHNICAL_CODING",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Your API's p95 latency spiked from 120ms to 900ms after deployment. What's your debugging approach?",
            "expected_depth": ["investigation", "analysis", "optimization"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "tech_ps_002",
            "role_category": "TECHNICAL_CODING",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "You need to optimize a slow database query affecting 1M+ records. Walk me through your process.",
            "expected_depth": ["indexing", "query_optimization", "architecture"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "tech_beh_001",
            "role_category": "TECHNICAL_CODING",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Tell me about a time you had to explain a complex technical concept to a non-technical stakeholder.",
            "expected_depth": ["communication", "simplification", "stakeholder_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # TECHNICAL_DATA scenarios
        {
            "scenario_id": "data_ps_001",
            "role_category": "TECHNICAL_DATA",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Your ML model's accuracy dropped from 95% to 78% in production. How would you investigate and fix this?",
            "expected_depth": ["model_drift", "data_analysis", "retraining"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "data_ps_002",
            "role_category": "TECHNICAL_DATA",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "You need to build a real-time recommendation system for 10M users. Describe your architecture and data pipeline.",
            "expected_depth": ["architecture", "scalability", "data_pipeline", "ml_ops"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "data_beh_001",
            "role_category": "TECHNICAL_DATA",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Describe a time when your data analysis led to a significant business decision. What was the impact?",
            "expected_depth": ["business_impact", "analysis", "communication"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # TECHNICAL_INFRASTRUCTURE scenarios
        {
            "scenario_id": "infra_ps_001",
            "role_category": "TECHNICAL_INFRASTRUCTURE",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Your production system is experiencing intermittent outages. Walk me through your troubleshooting process.",
            "expected_depth": ["monitoring", "debugging", "incident_response"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "infra_ps_002",
            "role_category": "TECHNICAL_INFRASTRUCTURE",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "Design a highly available system that can handle 100x traffic spikes. What's your approach?",
            "expected_depth": ["scalability", "reliability", "architecture", "monitoring"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "infra_beh_001",
            "role_category": "TECHNICAL_INFRASTRUCTURE",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Tell me about a time you had to coordinate with multiple teams during a major system migration.",
            "expected_depth": ["coordination", "communication", "project_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # TECHNICAL_DESIGN scenarios
        {
            "scenario_id": "design_ps_001",
            "role_category": "TECHNICAL_DESIGN",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Users are complaining about the checkout flow. How would you identify and fix the UX issues?",
            "expected_depth": ["user_research", "usability_testing", "design_iteration"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "design_ps_002",
            "role_category": "TECHNICAL_DESIGN",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "Design a mobile app for elderly users with accessibility needs. What considerations would you make?",
            "expected_depth": ["accessibility", "user_research", "design_principles", "usability"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "design_beh_001",
            "role_category": "TECHNICAL_DESIGN",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Describe a time when you had to defend your design decisions to stakeholders who disagreed.",
            "expected_depth": ["communication", "design_rationale", "stakeholder_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # BUSINESS_STRATEGIC scenarios
        {
            "scenario_id": "biz_ps_001",
            "role_category": "BUSINESS_STRATEGIC",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Our product's user retention dropped 30% this quarter. How would you analyze and address this?",
            "expected_depth": ["data_analysis", "user_research", "strategy", "metrics"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "biz_ps_002",
            "role_category": "BUSINESS_STRATEGIC",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "You need to enter a new market with a 50% budget cut. What's your go-to-market strategy?",
            "expected_depth": ["market_analysis", "strategy", "resource_allocation", "risk_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "biz_beh_001",
            "role_category": "BUSINESS_STRATEGIC",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Tell me about a time you had to influence a decision without having direct authority.",
            "expected_depth": ["influence", "communication", "stakeholder_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # CREATIVE_MARKETING scenarios
        {
            "scenario_id": "mkt_ps_001",
            "role_category": "CREATIVE_MARKETING",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Our campaign's conversion rate is 2% below target. How would you diagnose and improve it?",
            "expected_depth": ["analytics", "a_b_testing", "creative_optimization", "audience_targeting"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "mkt_ps_002",
            "role_category": "CREATIVE_MARKETING",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "Design a viral marketing campaign for a B2B SaaS product with a $10K budget.",
            "expected_depth": ["creative_strategy", "channel_selection", "budget_allocation", "measurement"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "mkt_beh_001",
            "role_category": "CREATIVE_MARKETING",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Describe a time when you had to pivot a marketing strategy mid-campaign due to poor performance.",
            "expected_depth": ["adaptability", "data_driven_decisions", "crisis_management"],
            "follow_up_triggers": ["deeper_probe"]
        },
        
        # SALES_BUSINESS scenarios
        {
            "scenario_id": "sales_ps_001",
            "role_category": "SALES_BUSINESS",
            "stage": "problem_solving",
            "difficulty": "intermediate",
            "question_text": "Your pipeline is 40% short of quota with 2 weeks left. What's your recovery strategy?",
            "expected_depth": ["pipeline_analysis", "prospecting", "deal_acceleration", "relationship_building"],
            "follow_up_triggers": ["deeper_probe", "escalation"]
        },
        {
            "scenario_id": "sales_ps_002",
            "role_category": "SALES_BUSINESS",
            "stage": "problem_solving",
            "difficulty": "advanced",
            "question_text": "A key prospect is stalling on a $500K deal. How would you handle this situation?",
            "expected_depth": ["objection_handling", "relationship_management", "negotiation", "value_proposition"],
            "follow_up_triggers": ["deeper_probe"]
        },
        {
            "scenario_id": "sales_beh_001",
            "role_category": "SALES_BUSINESS",
            "stage": "behavioral",
            "difficulty": "basic",
            "question_text": "Tell me about a time you lost a big deal. What did you learn from it?",
            "expected_depth": ["learning", "resilience", "self_reflection", "improvement"],
            "follow_up_triggers": ["deeper_probe"]
        }
    ]
    
    print("Seeding scenarios to ChromaDB...")
    for scenario in scenarios:
        try:
            interview_chroma.store_scenario_question(**scenario)
            print(f"✓ Stored: {scenario['scenario_id']} - {scenario['role_category']} {scenario['stage']}")
        except Exception as e:
            print(f"✗ Failed to store {scenario['scenario_id']}: {e}")
    
    print(f"\nSuccessfully seeded {len(scenarios)} scenarios")

if __name__ == "__main__":
    seed_scenarios()

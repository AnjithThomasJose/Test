"""
Mock response system for development environment.
Provides realistic mock data for all agent responses to avoid LLM calls during development.
"""

import asyncio
import json
import time
import uuid
import re
from typing import Dict, Any, List
from datetime import datetime
from urllib.parse import urlparse

# Mock data for different agent responses
MOCK_RESPONSES = {
    "validate_resume": True,
    "is_valid_resume": True,
    "validate_jd": {
        "is_valid_jd": True,
        "validation_message": "Job description text appears valid",
        "confidence_score": 0.85
    },

    # New unified resume parser (replaces personal_info_parser, education_parser, experience_parser, skills_parser)
    "groq_resume_parser": {
        "name": "John Doe",
        "professional_summary": "Full-stack engineer with 6+ years building scalable web apps.",
        "contact_details": {
            "Email": "john.doe@example.com",
            "Phone": "+1-555-0123",
            "Location": "San Francisco, CA",
            "LinkedIn": "linkedin.com/in/johndoe",
            "GitHub": "github.com/johndoe",
            "Website": ""
        },
        "education": [
            {
                "degree": "Bachelor of Technology",
                "major": "Computer Science",
                "university": "Stanford University",
                "location": "Stanford, CA",
                "years": "2015-2019",
                "gpa": "3.8/4.0",
                "capstone": {"title": "Scalable Microservices", "outcome": "Deployed to GCP"}
            }
        ],
        "work_experience": [
            {
                "job_title": "Senior Software Engineer",
                "company": "Tech Corp",
                "dates": "2021 - Present",
                "location": "San Francisco, CA",
                "responsibilities": [
                    "Led development of microservices architecture",
                    "Mentored junior developers",
                    "Improved system performance by 40%"
                ]
            },
            {
                "job_title": "Software Engineer",
                "company": "StartupXYZ",
                "dates": "2019 - 2021",
                "location": "Palo Alto, CA",
                "responsibilities": [
                    "Developed React-based frontend applications",
                    "Implemented RESTful APIs using Node.js",
                    "Collaborated with cross-functional teams"
                ]
            }
        ],
        "skills": [
            {"SkillName": "Python", "Proficiency": "", "Rationale": ""},
            {"SkillName": "JavaScript", "Proficiency": "", "Rationale": ""},
            {"SkillName": "React", "Proficiency": "", "Rationale": ""},
            {"SkillName": "AWS", "Proficiency": "", "Rationale": ""}
        ],
        "certifications": [
            {
                "certification_name": "AWS Certified Solutions Architect",
                "issuing_organization": "Amazon Web Services",
                "year": "2022"
            }
        ],
        "projects": [
            {
                "project_name": "E-commerce Platform",
                "description": "Full-stack e-commerce solution with React and Node.js",
                "technologies": ["React", "Node.js", "MongoDB", "AWS"],
                "duration": "6 months",
                "achievements": ["Launched MVP", "Handled 10k MAU"]
            },
            {
                "project_name": "AI Chatbot",
                "description": "Customer service chatbot using NLP and ML",
                "technologies": ["Python", "TensorFlow", "NLP"],
                "duration": "4 months",
                "achievements": ["Reduced support tickets by 25%"]
            }
        ],
        "professional_affiliations": [
            {"organization": "ACM", "membership": "Member", "years": "2020–Present"}
        ],
        "extras": [
            {"type": "Volunteer Work", "description": "Code for America volunteer", "duration": "2020-2021"}
        ],
        "total_experience_years": 5,
        "groq_parser_status": "success",
        "processing_time": 0.8,
        "method": "groq_single_call"
    },
    
    "resume_scorer": {
        "resumeScore": {
            "breakdown": {
                "experience": 85.0,
                "skills": 80.0,
                "education": 75.0,
                "presentation": 90.0
            },
            "total": 83.0,
            "rationale": "The candidate has a strong practical experience as a Full-Stack Developer and Co-Founder, demonstrating a good range of technical skills including Python, SQL, REST APIs, and cloud platforms like AWS. The projects showcase application of these skills in areas like machine learning, NLP, and web development. The education is relevant with a B.Tech in Computer Science. The resume is well-structured and easy to read, contributing to a good format score. Areas for improvement could include quantifying achievements further and potentially adding more diverse project experiences if applicable."
        }
    },
    
    "interest_filler": {
        "user_interests": [
            "I am naturally good at building intelligent systems using machine learning and computer vision, with proficiency in Python, TensorFlow, and full-stack development.",
            "I am actively seeking opportunities to contribute to innovative AI projects and grow within dynamic tech teams, implying a continuous learning approach in AI and related fields.",
            "My short-term aspiration is to contribute to innovative AI projects and grow within dynamic tech teams, leveraging my skills in machine learning and computer vision.",
            "In the long term, I aim to be a key contributor to groundbreaking AI projects, potentially leading initiatives and driving innovation in the field.",
            "Building intelligent systems and delivering production-ready AI tools and web platforms excites me the most, both in my academic work and professional experience.",
            "The provided resume does not contain information about my activities outside of work or studies.",
            "I am seeking opportunities to contribute to innovative AI projects and grow within dynamic tech teams, and JobsifyAI appears to be a platform that can facilitate such connections.",
            "I am most interested in roles focused on AI, machine learning, and computer vision, particularly within innovative tech companies.",
            "I have a LinkedIn profile and my resume is available, though specific links are not provided here.",
            "Yes, I am always open to mentorship and training suggestions that can help me grow and improve my skills in AI and computer science."
        ],
        "confidence_score": 0.8
    },
    
    "career_advisor": {
        "raw_skill_gap_analysis_output": {
            "career_paths": [
                {
                    "title": "AI/ML Engineer (Entry-Level)",
                    "description": "Focus on developing and deploying ML models.",
                    "required_skills": ["Python", "TensorFlow", "PyTorch", "SQL", "Cloud"]
                }
            ],
            "missing_skills": {"critical": ["Advanced NLP"], "high": ["Kubernetes"], "medium": []},
            "improvement_recommendations": [
                "Build a portfolio of AI/ML projects",
                "Deepen experience with ML frameworks"
            ],
            "salary_trends": {},
            "skill_demand_analysis": {}
        },
        "confidence_score": 0.8
    },
    
    "market_and_course_recommender": {
        "raw_skill_gap_analysis_output": {
            "salary_trends": {},
            "skill_demand_analysis": {}
        },
        "market_insights": [
            "Demand for AI/ML Engineers is growing across industries.",
            "Entry-level roles require Python, ML fundamentals, and practical projects.",
            "AI-focused software engineers integrate ML into scalable systems."
        ],
        "course_recommendations": [
            {
                "course": "Python for Everybody Specialization",
                "platform": "Coursera",
                "url": "https://www.coursera.org/specializations/python",
                "difficulty": "All Levels",
                "duration": "",
                "relevance_score": 0.9,
                "description": "Learn Python for real-world applications.",
                "url_validation": {"status": "valid", "accessible": True}
            },
            {
                "course": "Machine Learning",
                "platform": "Coursera",
                "url": "https://www.coursera.org/learn/machine-learning",
                "difficulty": "All Levels",
                "duration": "",
                "relevance_score": 0.95,
                "description": "Broad ML introduction.",
                "url_validation": {"status": "valid", "accessible": True}
            }
        ],
        "confidence_score": 0.8
    },
    
    "assessment_recommender": {
        "assessment_plan": [
            {
                "type": "mcq",
                "topic": "Psychometric Test",
                "num_questions": 10,
                "timer_per_question": 30,
                "difficulty": "Easy"
            },
            {
                "type": "mcq",
                "topic": "Communication Test",
                "num_questions": 10,
                "timer_per_question": 20,
                "difficulty": "Easy"
            },
            {
                "type": "mcq",
                "topic": "Personality Test",
                "num_questions": 10,
                "timer_per_question": 25,
                "difficulty": "Easy"
            },
            {
                "type": "multi",
                "topic": "Google",
                "difficulty": "Easy",
                "assessment_time_minutes": 46,
                "num_questions": {
                    "mcq": 5,
                    "short": 2,
                    "long": 1,
                    "coding": 1
                }
            }
        ],
        "confidence_score": 0.8
    },
    
    "assessment_question_generator": {
        "generated_questions": {
            "multi": [
                {
                    "question_text": "Which of the following is NOT a primitive data type in JavaScript?",
                    "options": {
                        "A": "String",
                        "B": "Number",
                        "C": "Array",
                        "D": "Boolean"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "What is the output of the following code snippet?\n\n```javascript\nlet x = 5;\nlet y = '5';\nconsole.log(x == y);\nconsole.log(x === y);\n```",
                    "options": {
                        "A": "true, true",
                        "B": "false, false",
                        "C": "true, false",
                        "D": "false, true"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Explain the concept of 'closures' in JavaScript and provide a practical example of its use.",
                    "question_type": "long_answer"
                },
                {
                    "question_text": "Describe the difference between `let`, `const`, and `var` in JavaScript, focusing on scope and mutability.",
                    "question_type": "short_answer"
                },
                {
                    "question_text": "Write a JavaScript function that takes an array of numbers and returns a new array containing only the even numbers from the original array. The function should be named `filterEvenNumbers` and accept one argument: `numbers` (an array of numbers).",
                    "language": "javascript",
                    "function_signature": "function filterEvenNumbers(numbers)",
                    "starter_code": "function filterEvenNumbers(numbers) {\n    // Your code here\n}",
                    "question_type": "coding"
                }
            ]
        },
        "total_questions": 5,
        "confidence_level": "high",
        "analysis_context": "question_generation",
        "processing_time_seconds": 2.5
    },
    
    "assessment_evaluator": {
        "assessment_topic": "JavaScript Fundamentals",
        "assessment_results": {
            "total_score": 85,
            "max_score": 100,
            "section_scores": {
                "mcq": 90,
                "short_answer": 80,
                "long_answer": 85
            },
            "question_scores": {
                "mcq_0": 10,
                "mcq_1": 10,
                "mcq_2": 10,
                "mcq_3": 10,
                "mcq_4": 10,
                "shortanswer0": 8,
                "shortanswer1": 8,
                "longanswer0": 9
            },
            "per_question": {
                "mcq": [
                    {
                        "question": "What is the correct way to create a React component?",
                        "user_answer": "D",
                        "evaluation": {
                            "score": 5,
                            "explanation": "Correct. All three methods are valid ways to create React components.",
                            "correct_answer": "All of the above"
                        }
                    },
                    {
                        "question": "Which of the following is NOT a primitive data type in JavaScript?",
                        "user_answer": "C",
                        "evaluation": {
                            "score": 5,
                            "explanation": "Correct. Array is not a primitive data type in JavaScript.",
                            "correct_answer": "Array"
                        }
                    }
                ],
                "short": [
                    {
                        "question": "What is the purpose of the useEffect hook in React?",
                        "user_answer": "The useEffect hook is used to perform side effects in functional components, such as data fetching, subscriptions, or manually changing the DOM.",
                        "evaluation": {
                            "score": 4,
                            "explanation": "Correct. useEffect is used for side effects in functional components.",
                            "correct_answer": "useEffect is used to perform side effects in functional components"
                        }
                    }
                ],
                "long": [
                    {
                        "question": "Explain the concept of closures in JavaScript with a practical example.",
                        "user_answer": "A closure is a function that has access to variables in its outer scope even after the outer function has returned. This is useful for creating private variables and maintaining state.",
                        "evaluation": {
                            "score": 4,
                            "explanation": "Good explanation of closures. Could benefit from a more detailed example.",
                            "correct_answer": "Closures allow functions to access variables from their outer scope even after the outer function has completed execution"
                        }
                    }
                ]
            }
        }
    },
    
    "report_generator": {
        "report": {
            "total_score": 85,
            "max_score": 100,
            "summary": "Demonstrated a good understanding, achieving 85.0%.",
            "performance_level": "Good",
            "topics": ["JavaScript", "React", "Algorithms"],
            "feedback": "Strong performance in JavaScript and React. Consider practicing more algorithmic problems to improve overall score.",
            "suggested_next_steps": [
                "Focus on data structures practice",
                "Review advanced React patterns",
                "Practice system design problems"
            ]
        }
    },
    
    "notification_agent": {
        "notification_sent": True,
        "message": "Assessment results have been sent to john.doe@example.com",
        "delivery_status": "Delivered",
        "timestamp": datetime.now().isoformat()
    },
    
    "job_description_parser": {
        "job_description": {
            "experience": "2+ years",
            "fileUrl": "https://firebasestorage.googleapis.com/v0/b/jobsify-5d910.firebasestorage.app/o/job_description%2FWr9L1LAcuqaZJzww1IEd3y5DCBe2%2FlbqnVXjSM3HbScCSrgub%2FFrontend_Developer_Job_Description_With_Salary.docx?alt=media&token=0059f12a-ebd6-4440-b078-789af49fecb1",
            "fullJobDescription": "Job Description Job Title: Frontend Developer Location: Chennai / Remote Employment Type: Full-time About the Company At TechNova Solutions, we build scalable digital products that transform businesses. We work with global clients across e-commerce, fintech, healthcare, and SaaS to deliver high-quality web and mobile applications. Our team thrives on innovation, collaboration, and continuous learning. Job Description We are looking for a passionate Frontend Developer to join our engineering team. The ideal candidate should have strong experience in building modern, responsive web applications and a solid understanding of React.js and related technologies. Responsibilities Develop, test, and maintain high-quality web applications using React.js, HTML, CSS, and JavaScript. Collaborate with backend developers, designers, and product managers to deliver seamless user experiences. Implement responsive design and ensure cross-browser compatibility. Optimize applications for maximum speed, scalability, and performance. Participate in code reviews, contribute to best practices, and maintain clean, reusable code. Stay updated with the latest industry trends and emerging frontend technologies. Requirements Bachelor's degree in Computer Science, Engineering, or related field (or equivalent experience). 2+ years of experience in frontend development. Strong proficiency in JavaScript (ES6+), HTML5, CSS3. Hands-on experience with React.js, Redux, Ant Design (or similar UI libraries). Familiarity with RESTful APIs, Git, and agile methodologies. Good understanding of web performance optimization and security practices. Excellent problem-solving skills and attention to detail. Required Skills Proficiency in React.js, Redux, and modern JavaScript frameworks. Strong knowledge of responsive design principles. Experience with version control systems (Git). Ability to write clean, maintainable, and scalable code. Good communication and teamwork skills. Experience Minimum 2 years of professional frontend development experience. Proven track record of delivering web applications in production environments. Experience in collaborating with cross-functional teams. Compensation 600000 - 1200000 per annum (based on experience and skills). Nice to Have Experience with TypeScript, Next.js, or Vue.js. Knowledge...",
            "jobTitle": "passionate Frontend Developer",
            "jobType": "full-time, remote",
            "location": "HTML, CSS",
            "requiredSkills": [
                "React.js",
                "HTML5",
                "CSS3",
                "JavaScript",
                "Redux",
                "Ant Design",
                "RESTful APIs",
                "Git",
                "Agile",
                "Web Performance Optimization",
                "Security Practices",
                "Responsive Design",
                "TypeScript",
                "Next.js",
                "Vue.js",
                "Jest",
                "Cypress",
                "CI/CD",
                "AWS",
                "GCP",
                "Azure"
            ],
            "salary": "600000 - 1200000 per annum"
        },
        "confidence_score": 0.95,
        "confidence_level": "high",
        "processing_time_seconds": 2.1
    },
    
    "ranker": {
        "ranked_candidates": [
            {
                "rank": 1,
                "candidate_id": "mock_candidate_001",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+91-9876543210",
                "match_score": 0.85,
                "skills_matched": [
                    "React.js",
                    "HTML5",
                    "CSS3",
                    "JavaScript",
                    "Redux",
                    "RESTful APIs",
                    "Git",
                    "Agile",
                    "Web Performance Optimization",
                    "Responsive Design",
                    "TypeScript",
                    "Next.js",
                    "Vue.js",
                    "Jest",
                    "Cypress"
                ],
                "skills_unmatched": [
                    "Ant Design",
                    "Security Practices",
                    "CI/CD",
                    "AWS",
                    "GCP",
                    "Azure"
                ],
                "rationale": "This candidate is an excellent match for the Frontend Developer role, demonstrating extensive experience with core requirements like React.js, Redux, and responsive design, along with valuable 'nice-to-have' skills such as TypeScript, Next.js, and Jest. Their proven track record in performance optimization, API integration, and agile methodologies makes them highly suitable for the position."
            },
            {
                "rank": 2,
                "candidate_id": "mock_candidate_002",
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+91-9876543211",
                "match_score": 0.72,
                "skills_matched": [
                    "React.js",
                    "HTML5",
                    "CSS3",
                    "JavaScript",
                    "RESTful APIs",
                    "Git",
                    "Agile",
                    "Responsive Design",
                    "TypeScript"
                ],
                "skills_unmatched": [
                    "Redux",
                    "Ant Design",
                    "Security Practices",
                    "Next.js",
                    "Vue.js",
                    "Jest",
                    "Cypress",
                    "CI/CD",
                    "AWS",
                    "GCP",
                    "Azure"
                ],
                "rationale": "This candidate shows strong potential with good experience in React.js and core web technologies. While they lack some advanced skills like Redux and testing frameworks, their solid foundation in frontend development and willingness to learn makes them a good fit for the role."
            },
            {
                "rank": 3,
                "candidate_id": "mock_candidate_003",
                "name": "Mike Johnson",
                "email": "mike.johnson@example.com",
                "phone": "+91-9876543212",
                "match_score": 0.65,
                "skills_matched": [
                    "HTML5",
                    "CSS3",
                    "JavaScript",
                    "RESTful APIs",
                    "Git",
                    "Agile",
                    "Web Performance Optimization",
                    "Responsive Design",
                    "AWS"
                ],
                "skills_unmatched": [
                    "React.js",
                    "Redux",
                    "Ant Design",
                    "Security Practices",
                    "TypeScript",
                    "Next.js",
                    "Vue.js",
                    "Jest",
                    "Cypress",
                    "CI/CD",
                    "GCP",
                    "Azure"
                ],
                "rationale": "This candidate has good foundational skills in web development but lacks experience with React.js and Redux, which are central to this role. Their experience with performance optimization and cloud platforms is valuable, but they would need significant training in React ecosystem."
            },
            {
                "rank": 4,
                "candidate_id": "mock_candidate_004",
                "name": "Sarah Wilson",
                "email": "sarah.wilson@example.com",
                "phone": "+91-9876543213",
                "match_score": 0.58,
                "skills_matched": [
                    "JavaScript",
                    "HTML5",
                    "CSS3",
                    "RESTful APIs",
                    "Git",
                    "Agile",
                    "Responsive Design"
                ],
                "skills_unmatched": [
                    "React.js",
                    "Redux",
                    "Ant Design",
                    "Security Practices",
                    "Web Performance Optimization",
                    "TypeScript",
                    "Next.js",
                    "Vue.js",
                    "Jest",
                    "Cypress",
                    "CI/CD",
                    "AWS",
                    "GCP",
                    "Azure"
                ],
                "rationale": "This candidate has basic web development skills but lacks the specific React.js experience required for this role. They would need extensive training in modern frontend frameworks and tools to be effective in this position."
            },
            {
                "rank": 5,
                "candidate_id": "mock_candidate_005",
                "name": "David Brown",
                "email": "david.brown@example.com",
                "phone": "+91-9876543214",
                "match_score": 0.45,
                "skills_matched": [
                    "HTML5",
                    "CSS3",
                    "Git",
                    "Agile"
                ],
                "skills_unmatched": [
                    "React.js",
                    "Redux",
                    "Ant Design",
                    "Security Practices",
                    "Web Performance Optimization",
                    "Responsive Design",
                    "TypeScript",
                    "Next.js",
                    "Vue.js",
                    "Jest",
                    "Cypress",
                    "CI/CD",
                    "AWS",
                    "GCP",
                    "Azure",
                    "JavaScript",
                    "RESTful APIs"
                ],
                "rationale": "This candidate has very limited frontend development experience and lacks most of the required skills for this role. They would need comprehensive training in JavaScript, React.js, and modern web development practices."
            }
        ],
        "confidence_score": 0.88,
        "confidence_level": "high",
        "processing_time_seconds": 3.2
    }
}

# Mock response timing (simulate real processing delays)
MOCK_DELAYS = {
    "validate_resume": 0.5,
    "groq_resume_parser": 1.2,
    "resume_scorer": 1.0,
    "interest_filler": 0.7,
    "career_advisor": 1.0,
    "market_and_course_recommender": 1.2,
    "assessment_recommender": 0.9,
    "assessment_question_generator": 1.2,
    "assessment_evaluator": 1.0,
    "report_generator": 0.9,
    "notification_agent": 0.3,
    "validate_jd": 0.4,
    "job_description_parser": 0.8,
    "ranker": 1.0
}

class MockResponseGenerator:
    """Generates mock responses for development environment."""
    
    def __init__(self):
        self.responses = MOCK_RESPONSES
        self.delays = MOCK_DELAYS
    
    def _get_assessment_id(self, state: Dict[str, Any]) -> str:
        """Get assessment_id using the same logic as the real assessment_question_generator."""
        # Same logic as: state.get("assessment_id", str(uuid.uuid4()))
        return state.get("assessment_id", str(uuid.uuid4()))
    
    def _get_question_doc_id(self, state: Dict[str, Any]) -> str:
        """Get question_doc_id using similar logic."""
        # Generate a related question doc ID based on assessment_id
        assessment_id = self._get_assessment_id(state)
        return f"questions_{assessment_id}"
    
    def _get_job_id(self, state: Dict[str, Any]) -> str:
        """Get job_id using the same logic as assessment_id."""
        # Same logic as: state.get("job_id", str(uuid.uuid4()))
        return state.get("job_id", str(uuid.uuid4()))
    
    def _validate_resume_url(self, resume_url: str) -> Dict[str, Any]:
        """Validate resume URL and determine if it's valid."""
        if not resume_url:
            return {
                "is_valid_resume": False,
                "validation_message": "No resume URL provided",
                "confidence_score": 0.0
            }
        
        try:
            parsed_url = urlparse(resume_url)
            
            # Check if it's a valid URL
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "is_valid_resume": False,
                    "validation_message": "Invalid URL format",
                    "confidence_score": 0.0
                }
            
            # Check for common resume file extensions
            valid_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
            path_lower = parsed_url.path.lower()
            
            if not any(path_lower.endswith(ext) for ext in valid_extensions):
                return {
                    "is_valid_resume": False,
                    "validation_message": "File format not supported. Please upload PDF, DOC, DOCX, TXT, or RTF files",
                    "confidence_score": 0.0
                }
            
            # Check for common resume hosting domains
            valid_domains = [
                'linkedin.com', 'indeed.com', 'monster.com', 'glassdoor.com',
                'dropbox.com', 'drive.google.com', 'onedrive.live.com',
                'github.com', 'gitlab.com', 'bitbucket.org'
            ]
            
            domain = parsed_url.netloc.lower()
            is_known_domain = any(valid_domain in domain for valid_domain in valid_domains)
            
            # Check for mock/test URLs
            is_mock_url = any(keyword in resume_url.lower() for keyword in [
                'mock', 'test', 'sample', 'example', 'demo', 'placeholder'
            ])
            
            if is_mock_url:
                return {
                    "is_valid_resume": True,
                    "validation_message": "Mock resume URL detected - using sample data",
                    "confidence_score": 0.9
                }
            elif is_known_domain:
                return {
                    "is_valid_resume": True,
                    "validation_message": "Resume URL appears valid and accessible",
                    "confidence_score": 0.85
                }
            else:
                return {
                    "is_valid_resume": True,
                    "validation_message": "Resume URL format is valid",
                    "confidence_score": 0.7
                }
                
        except Exception as e:
            return {
                "is_valid_resume": False,
                "validation_message": f"Error validating resume URL: {str(e)}",
                "confidence_score": 0.0
            }
    
    def _validate_jd_text(self, jd_text: str) -> Dict[str, Any]:
        """Validate job description text and determine if it's valid."""
        if not jd_text:
            return {
                "is_valid_jd": False,
                "validation_message": "No job description text provided",
                "confidence_score": 0.0
            }
        
        try:
            text = jd_text.lower()
            has_role = any(k in text for k in ["developer", "engineer", "manager", "designer", "analyst"])
            has_requirements = any(k in text for k in ["requirements", "responsibilities", "skills", "qualifications"])
            is_mock = any(k in text for k in ["mock", "sample", "example", "demo", "placeholder"])

            if is_mock:
                return {
                    "is_valid_jd": True,
                    "validation_message": "Mock job description text detected - using sample data",
                    "confidence_score": 0.9
                }
            elif has_role and has_requirements:
                return {
                    "is_valid_jd": True,
                    "validation_message": "Job description text appears valid",
                    "confidence_score": 0.85
                }
            else:
                return {
                    "is_valid_jd": True,
                    "validation_message": "Job description text format is acceptable",
                    "confidence_score": 0.6
                }
                
        except Exception as e:
            return {
                "is_valid_jd": False,
                "validation_message": f"Error validating job description text: {str(e)}",
                "confidence_score": 0.0
            }
    
    def _get_assessment_topic(self, assessment_plan: Dict[str, Any]) -> str:
        """Extract assessment topic from assessment plan."""
        if isinstance(assessment_plan, dict):
            return assessment_plan.get("topic", "General Assessment")
        elif isinstance(assessment_plan, list) and assessment_plan:
            return assessment_plan[0].get("topic", "General Assessment")
        return "General Assessment"
    
    def _generate_topic_based_questions(self, assessment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate questions based on assessment recommender topics."""
        # Extract topics from assessment plan
        topics = []
        if isinstance(assessment_plan, list):
            topics = [item.get("topic", "") for item in assessment_plan if item.get("topic")]
        elif isinstance(assessment_plan, dict):
            topic = assessment_plan.get("topic", "")
            if topic:
                topics = [topic]
        
        # Default to JavaScript if no topics found
        if not topics:
            topics = ["JavaScript"]
        
        # Generate questions based on the first topic
        primary_topic = topics[0].lower()
        
        if "psychometric" in primary_topic:
            return self._get_psychometric_questions()
        elif "communication" in primary_topic:
            return self._get_communication_questions()
        elif "personality" in primary_topic:
            return self._get_personality_questions()
        elif "google" in primary_topic:
            return self._get_google_questions()
        elif "javascript" in primary_topic:
            return self._get_javascript_questions()
        elif "python" in primary_topic:
            return self._get_python_questions()
        elif "react" in primary_topic:
            return self._get_react_questions()
        else:
            return self._get_javascript_questions()  # Default fallback
    
    def _get_psychometric_questions(self) -> Dict[str, Any]:
        """Generate psychometric test questions."""
        return {
            "multi": [
                {
                    "question_text": "When faced with a difficult problem, I prefer to:",
                    "options": {
                        "A": "Work on it alone first",
                        "B": "Discuss it with colleagues immediately",
                        "C": "Research similar problems online",
                        "D": "Ask for help from a supervisor"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "I am most motivated when:",
                    "options": {
                        "A": "Working on challenging projects",
                        "B": "Collaborating with a team",
                        "C": "Learning new technologies",
                        "D": "Meeting deadlines"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Describe a situation where you had to adapt to a significant change in your work environment.",
                    "question_type": "long_answer"
                }
            ]
        }
    
    def _get_communication_questions(self) -> Dict[str, Any]:
        """Generate communication test questions."""
        return {
            "multi": [
                {
                    "question_text": "How would you explain a complex technical concept to a non-technical stakeholder?",
                    "options": {
                        "A": "Use technical jargon to sound professional",
                        "B": "Use simple analogies and visual aids",
                        "C": "Provide detailed documentation",
                        "D": "Ask them to research it themselves"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "In a team meeting, how do you handle conflicting opinions?",
                    "options": {
                        "A": "Always support the majority view",
                        "B": "Facilitate discussion to find common ground",
                        "C": "Present your own solution forcefully",
                        "D": "Avoid taking sides"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Write a professional email to a client explaining a project delay.",
                    "question_type": "short_answer"
                }
            ]
        }
    
    def _get_personality_questions(self) -> Dict[str, Any]:
        """Generate personality test questions."""
        return {
            "multi": [
                {
                    "question_text": "I prefer working:",
                    "options": {
                        "A": "Independently on focused tasks",
                        "B": "In collaborative team environments",
                        "C": "On multiple projects simultaneously",
                        "D": "With clear, structured guidelines"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "My ideal work environment includes:",
                    "options": {
                        "A": "Quiet, distraction-free space",
                        "B": "Open, collaborative workspace",
                        "C": "Flexible, remote options",
                        "D": "Traditional office setting"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Describe your approach to handling stress and pressure in the workplace.",
                    "question_type": "long_answer"
                }
            ]
        }
    
    def _get_google_questions(self) -> Dict[str, Any]:
        """Generate Google-style technical questions."""
        return {
            "multi": [
                {
                    "question_text": "Design a system to handle 1 million requests per second. What are the key considerations?",
                    "question_type": "long_answer"
                },
                {
                    "question_text": "Implement a function to find the longest common subsequence between two strings.",
                    "language": "python",
                    "function_signature": "def longest_common_subsequence(text1, text2)",
                    "starter_code": "def longest_common_subsequence(text1, text2):\n    # Your code here\n    pass",
                    "question_type": "coding"
                },
                {
                    "question_text": "What is the time complexity of quicksort in the worst case?",
                    "options": {
                        "A": "O(n)",
                        "B": "O(n log n)",
                        "C": "O(n²)",
                        "D": "O(log n)"
                    },
                    "question_type": "mcq"
                }
            ]
        }
    
    def _get_javascript_questions(self) -> Dict[str, Any]:
        """Generate JavaScript questions."""
        return {
            "multi": [
                {
                    "question_text": "Which of the following is NOT a primitive data type in JavaScript?",
                    "options": {
                        "A": "String",
                        "B": "Number",
                        "C": "Array",
                        "D": "Boolean"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "What is the output of the following code snippet?\n\n```javascript\nlet x = 5;\nlet y = '5';\nconsole.log(x == y);\nconsole.log(x === y);\n```",
                    "options": {
                        "A": "true, true",
                        "B": "false, false",
                        "C": "true, false",
                        "D": "false, true"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Explain the concept of 'closures' in JavaScript and provide a practical example.",
                    "question_type": "long_answer"
                },
                {
                    "question_text": "Write a JavaScript function that takes an array of numbers and returns a new array containing only the even numbers.",
                    "language": "javascript",
                    "function_signature": "function filterEvenNumbers(numbers)",
                    "starter_code": "function filterEvenNumbers(numbers) {\n    // Your code here\n}",
                    "question_type": "coding"
                }
            ]
        }
    
    def _get_python_questions(self) -> Dict[str, Any]:
        """Generate Python questions."""
        return {
            "multi": [
                {
                    "question_text": "What is the output of the following Python code?\n\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)\n```",
                    "options": {
                        "A": "[1, 2, 3]",
                        "B": "[1, 2, 3, 4]",
                        "C": "[4]",
                        "D": "Error"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Which of the following is NOT a Python data structure?",
                    "options": {
                        "A": "List",
                        "B": "Dictionary",
                        "C": "Array",
                        "D": "Tuple"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Explain the difference between 'is' and '==' operators in Python.",
                    "question_type": "short_answer"
                },
                {
                    "question_text": "Write a Python function to reverse a string without using built-in reverse methods.",
                    "language": "python",
                    "function_signature": "def reverse_string(s)",
                    "starter_code": "def reverse_string(s):\n    # Your code here\n    pass",
                    "question_type": "coding"
                }
            ]
        }
    
    def _get_react_questions(self) -> Dict[str, Any]:
        """Generate React questions."""
        return {
            "multi": [
                {
                    "question_text": "What is the correct way to create a React component?",
                    "options": {
                        "A": "function MyComponent() { return <div>Hello</div>; }",
                        "B": "class MyComponent extends React.Component { render() { return <div>Hello</div>; } }",
                        "C": "const MyComponent = () => <div>Hello</div>;",
                        "D": "All of the above"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "What is the purpose of the useEffect hook in React?",
                    "options": {
                        "A": "To manage component state",
                        "B": "To perform side effects",
                        "C": "To handle events",
                        "D": "To create components"
                    },
                    "question_type": "mcq"
                },
                {
                    "question_text": "Explain the concept of 'props' in React and how they differ from state.",
                    "question_type": "long_answer"
                },
                {
                    "question_text": "Create a React component that displays a counter with increment and decrement buttons.",
                    "language": "javascript",
                    "function_signature": "function Counter()",
                    "starter_code": "function Counter() {\n    // Your code here\n}",
                    "question_type": "coding"
                }
            ]
        }
    
    async def get_mock_response(self, node_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get mock response for a specific node.
        
        Args:
            node_name: Name of the agent/node
            state: Current state (for context-aware responses)
            
        Returns:
            Mock response data
        """
        # Simulate processing delay
        delay = self.delays.get(node_name, 1.0)
        await asyncio.sleep(delay)
        
        # Get base response
        base_response = self.responses.get(node_name, {})
        
        # Customize response based on state if needed
        customized_response = self._customize_response(node_name, base_response, state)
        
        return customized_response
    
    def _customize_response(self, node_name: str, base_response: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Customize mock response based on current state."""
        # Handle special case for validate_resume - always validate and return boolean
        if node_name == "validate_resume":
            body = state.get("body", {})
            resume_url = body.get("resume_url", "")
            validation_result = self._validate_resume_url(resume_url)
            return validation_result.get("is_valid_resume", True)
        
        # Handle special case where base_response is not a dict (e.g., other boolean responses)
        if not isinstance(base_response, dict):
            return base_response
        
        customized = base_response.copy()
        
        if node_name == "validate_jd":
            # Validate JD text and return appropriate response
            body = state.get("body", {})
            jd_text = body.get("jd_text") or body.get("job_description_text") or ""
            if not jd_text and body.get("jd_url"):
                # If only URL is provided in dev, accept as valid
                validation_result = {
                    "is_valid_jd": True,
                    "validation_message": "JD URL provided; treating as valid in mock",
                    "confidence_score": 0.8
                }
            else:
                validation_result = self._validate_jd_text(jd_text)
            customized.update(validation_result)
            # Add job_id using the same logic as assessment_id
            customized["job_id"] = self._get_job_id(state)
        
        elif node_name == "personal_info_parser":
            # Check if resume is valid before proceeding
            body = state.get("body", {})
            resume_url = body.get("resume_url", "")
            resume_validation = self._validate_resume_url(resume_url)
            
            if not resume_validation.get("is_valid_resume", False):
                # Return error response for invalid resume
                customized = {
                    "error": "Invalid resume URL",
                    "message": resume_validation.get("validation_message", "Resume validation failed"),
                    "confidence_score": 0.0
                }
            else:
                # Use actual name from resume if available
                resume_data = state.get("structured_resume", {})
                if resume_data and "name" in resume_data:
                    customized["name"] = resume_data["name"]
        
        elif node_name == "education_parser":
            # Check if resume is valid before proceeding
            body = state.get("body", {})
            resume_url = body.get("resume_url", "")
            resume_validation = self._validate_resume_url(resume_url)
            
            if not resume_validation.get("is_valid_resume", False):
                # Return error response for invalid resume
                customized = {
                    "error": "Invalid resume URL",
                    "message": resume_validation.get("validation_message", "Resume validation failed"),
                    "confidence_score": 0.0
                }
        
        elif node_name == "experience_parser":
            # Check if resume is valid before proceeding
            body = state.get("body", {})
            resume_url = body.get("resume_url", "")
            resume_validation = self._validate_resume_url(resume_url)
            
            if not resume_validation.get("is_valid_resume", False):
                # Return error response for invalid resume
                customized = {
                    "error": "Invalid resume URL",
                    "message": resume_validation.get("validation_message", "Resume validation failed"),
                    "confidence_score": 0.0
                }
        
        elif node_name == "job_description_parser":
            # Check if JD is valid before proceeding
            body = state.get("body", {})
            jd_text = body.get("jd_text") or body.get("job_description_text") or ""
            jd_validation = self._validate_jd_text(jd_text) if jd_text else {"is_valid_jd": True}
            if not jd_validation.get("is_valid_jd", False):
                customized = {
                    "error": "Invalid job description",
                    "message": jd_validation.get("validation_message", "Job description validation failed"),
                    "confidence_score": 0.0
                }
            else:
                customized["job_id"] = self._get_job_id(state)
        
        elif node_name == "assessment_question_generator":
            # Use the SAME logic as the real assessment_question_generator
            customized["assessment_id"] = self._get_assessment_id(state)
            
            # Generate questions based on assessment recommender topics
            assessment_plan = state.get("assessment_plan", {})
            if assessment_plan:
                customized["generated_questions"] = self._generate_topic_based_questions(assessment_plan)
        
        elif node_name == "assessment_evaluator":
            # Use consistent IDs for the session
            customized["assessment_id"] = self._get_assessment_id(state)
            customized["question_doc_id"] = self._get_question_doc_id(state)
            
            # Set assessment topic based on assessment plan
            assessment_plan = state.get("assessment_plan", {})
            if assessment_plan:
                customized["assessment_topic"] = self._get_assessment_topic(assessment_plan)
        
        elif node_name == "report_generator":
            # Use consistent IDs for the session
            customized["assessment_id"] = self._get_assessment_id(state)
            customized["question_doc_id"] = self._get_question_doc_id(state)
        
        return customized
    
    def get_mock_stream_events(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a sequence of mock streaming events based on the request type.
        
        Args:
            state: Request state to determine which agents to simulate
            
        Returns:
            List of mock streaming events
        """
        events = []
        body = state.get("body", {})
        
        # Determine which agents to simulate based on request type
        agents_to_simulate = self._determine_agents_to_simulate(body)
        
        # Generate consistent assessment IDs for this session
        session_assessment_id = self._get_assessment_id(state)
        session_question_doc_id = f"questions_{session_assessment_id}"
        
        # Add the assessment IDs to state for consistent usage across agents
        state_with_ids = state.copy()
        state_with_ids["assessment_id"] = session_assessment_id
        state_with_ids["question_doc_id"] = session_question_doc_id
        
        # Generate events for each agent
        for i, agent_name in enumerate(agents_to_simulate):
            if agent_name in self.responses:
                # Create mock node output with customization
                mock_output = self._customize_response(agent_name, self.responses[agent_name], state_with_ids)
                
                # Create streaming event (matching real LLM output format)
                event = {
                    agent_name: mock_output
                }
                events.append(event)
        
        return events
    
    def _determine_agents_to_simulate(self, body: Dict[str, Any]) -> List[str]:
        """Determine which agents to simulate based on request body and validation results."""
        agents = []
        
        # Always start with dispatcher
        agents.append("dispatcher")
        
        # Resume flow with validation
        if body.get("resume_url"):
            agents.append("validate_resume")
            
            # Check if resume is valid before proceeding with parsing
            resume_validation = self._validate_resume_url(body.get("resume_url", ""))
            if resume_validation.get("is_valid_resume", False):
                # Only add parsing agents if resume is valid
                agents.extend([
                    "groq_resume_parser",
                    "resume_scorer",
                    "interest_filler",
                    "career_advisor",
                    "market_and_course_recommender",
                    "assessment_recommender"
                ])
            # If resume is invalid, don't add any parsing agents - just validate_resume will return false
        
        # JD flow with validation
        if body.get("jd_url"):
            agents.append("validate_jd")
            
            # Check if JD is valid before proceeding with parsing (treat URL as valid in mock)
            jd_validation = {"is_valid_jd": True}
            if jd_validation.get("is_valid_jd", False):
                # Only add parsing agents if JD is valid
                agents.extend([
                    "job_description_parser",
                    "ranker"
                ])
            # If JD is invalid, don't add any parsing agents - just validate_jd will return false
        
        # Assessment question generation
        if body.get("plan"):
            agents.append("assessment_question_generator")
        
        # Assessment evaluation
        if body.get("submission"):
            agents.append("assessment_evaluator")
            agents.append("report_generator")
        
        # Notification
        if body.get("user_mail") or body.get("email"):
            agents.append("notification_agent")
        
        return agents

# Global instance
mock_generator = MockResponseGenerator()

import asyncio
import json
import logging
import re
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from langsmith.run_helpers import traceable

from core.model_registry import TaskType
from core.utils import _extract_json_from_response, _calculate_processing_time
from core.logging_helpers import create_log_context, log_agent_completion
from models.llm_invoker import invoke_llm

log = logging.getLogger(__name__)

TARGET_TOTAL_MARKS = 100
MAX_INPUT_LENGTH = 12000


def _detect_sections_from_text(raw_text: str) -> List[str]:
    """
    Automatically detect topics/sections from raw assessment text.
    
    Args:
        raw_text: Raw assessment description
        
    Returns:
        List of detected section names
    """
    import re
    
    # Common section indicators
    section_patterns = [
        r'(?i)chapter\s+\d+[:\s]*([^\n]+)',
        r'(?i)section\s+\d+[:\s]*([^\n]+)',
        r'(?i)topic\s+\d+[:\s]*([^\n]+)',
        r'(?i)module\s+\d+[:\s]*([^\n]+)',
        r'(?i)unit\s+\d+[:\s]*([^\n]+)',
        r'(?i)^\d+\.\s*([^\n:]+)',
        r'(?i)^\*\s*([^\n:]+)',
        r'(?i)^-\s*([^\n:]+)'
    ]
    
    sections = []
    for pattern in section_patterns:
        matches = re.findall(pattern, raw_text, re.MULTILINE)
        for match in matches:
            section_name = match.strip()
            if section_name and len(section_name) > 3:  # Filter out very short matches
                sections.append(section_name)
    
    # If no clear sections found, create logical groupings
    if not sections:
        # Look for topic keywords in the text
        topic_keywords = {
            'Programming': ['programming', 'coding', 'development', 'software', 'code'],
            'Database': ['database', 'sql', 'data', 'storage', 'query'],
            'Web Development': ['web', 'html', 'css', 'javascript', 'frontend'],
            'Mathematics': ['math', 'calculation', 'arithmetic', 'algebra', 'geometry'],
            'Logic': ['logic', 'reasoning', 'problem solving', 'analytical'],
            'Theory': ['theory', 'concept', 'fundamental', 'basic', 'introduction']
        }
        
        text_lower = raw_text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sections.append(topic)
    
    # Fallback to generic sections
    if not sections:
        sections = ['General Assessment']
    
    return sections[:5]  # Limit to 5 sections


def _extract_questions_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract individual questions from raw assessment text.
    
    Args:
        raw_text: Raw assessment description
        
    Returns:
        List of question dictionaries
    """
    import re
    
    questions = []
    
    # First, split by common sentence separators
    sentences = re.split(r'[.!?]+', raw_text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip very short sentences
        if len(sentence) < 10:
            continue
            
        # Check if it's already a question (ends with ?)
        if sentence.endswith('?'):
            questions.append({
                'question': sentence,
                'original_text': sentence
            })
            continue
        
        # Check if it contains question keywords
        question_keywords = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'describe', 'explain', 'define', 'calculate']
        if any(keyword in sentence.lower() for keyword in question_keywords):
            # Convert to question if not already
            if not sentence.endswith('?'):
                sentence = sentence + '?'
            questions.append({
                'question': sentence,
                'original_text': sentence
            })
            continue
        
        # Check for numbered questions
        numbered_match = re.match(r'^\d+\.\s*(.+)', sentence)
        if numbered_match:
            question_text = numbered_match.group(1).strip()
            if not question_text.endswith('?'):
                question_text = question_text + '?'
            questions.append({
                'question': question_text,
                'original_text': question_text
            })
    
    # Remove duplicates while preserving order
    seen_questions = set()
    unique_questions = []
    for q in questions:
        question_lower = q['question'].lower()
        if question_lower not in seen_questions:
            seen_questions.add(question_lower)
            unique_questions.append(q)
    
    return unique_questions[:20]  # Limit to 20 questions


def _infer_difficulty(question_text: str) -> str:
    """
    Infer difficulty level from question text.
    
    Args:
        question_text: The question text
        
    Returns:
        Difficulty level: 'easy', 'medium', or 'hard'
    """
    text_lower = question_text.lower()
    
    # Easy indicators
    easy_keywords = ['basic', 'simple', 'what is', 'define', 'identify', 'list', 'name']
    
    # Hard indicators
    hard_keywords = ['complex', 'analyze', 'evaluate', 'compare', 'contrast', 'design', 'implement', 'optimize']
    
    if any(keyword in text_lower for keyword in easy_keywords):
        return 'easy'
    elif any(keyword in text_lower for keyword in hard_keywords):
        return 'hard'
    else:
        return 'medium'


def _generate_mcq_options(question_text: str, difficulty: str) -> List[str]:
    """
    Generate 4 real MCQ options for a given question.
    
    Args:
        question_text: The question text
        difficulty: Difficulty level
        
    Returns:
        List of 4 real options with one being correct
    """
    import re
    
    question_lower = question_text.lower()
    
    # Programming questions FIRST - expanded keywords
    if any(word in question_lower for word in [
        'programming', 'code', 'function', 'variable', 'loop', 'array', 
        'declare', 'inheritance', 'polymorphism', 'encapsulation', 'object-oriented',
        'class', 'method', 'algorithm', 'syntax', 'debug'
    ]):
        return _generate_programming_options(question_text)
    
    # Database questions
    elif any(word in question_lower for word in ['database', 'sql', 'table', 'query', 'primary key', 'foreign key']):
        return _generate_database_options(question_text)
    
    # Web development questions
    elif any(word in question_lower for word in ['html', 'css', 'javascript', 'web', 'frontend', 'responsive']):
        return _generate_web_options(question_text)
    
    # Object-oriented programming
    elif any(word in question_lower for word in ['object', 'class', 'inheritance', 'polymorphism', 'encapsulation', 'oops']):
        return _generate_oop_options(question_text)
    
    # Math questions LAST - only if math-specific patterns and no programming keywords
    math_keywords = [
        'calculate', 'find', 'solve', 'compute',
        'square root', 'percent', 'divided', 'volume', 'perimeter', 'area'
    ]
    has_math_keyword = any(w in question_lower for w in math_keywords)
    has_math_op = any(op in question_text for op in ['+', '-', '×', '*', '/', 'plus', 'minus', 'multiply', 'divide'])
    has_percent = '%' in question_text
    has_digits = bool(re.search(r'\d+', question_text))
    if (has_math_keyword or has_math_op or has_percent) and has_digits:
        return _generate_math_options(question_text)
    
    # General/concept questions
    else:
        return _generate_general_options(question_text)


def _generate_math_options(question_text: str) -> List[str]:
    """Generate real math options with calculated correct answers"""
    import random
    
    # Extract numbers and operations
    question_lower = question_text.lower()
    
    # Basic arithmetic
    if '+' in question_text and 'what is' in question_lower:
        # Extract numbers for addition
        numbers = re.findall(r'\d+', question_text)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            correct = a + b
            return [
                str(correct),
                str(correct + random.randint(1, 10)),
                str(correct - random.randint(1, 5) if correct > 5 else 1),
                str(correct + random.randint(5, 15))
            ]
    
    elif '×' in question_text or 'multiply' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            correct = a * b
            return [
                str(correct),
                str(correct + random.randint(1, 20)),
                str(correct - random.randint(1, 10) if correct > 10 else 1),
                str(a + b)  # Common wrong answer (addition instead of multiplication)
            ]
    
    elif 'area' in question_lower and 'rectangle' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if len(numbers) >= 2:
            length, width = int(numbers[0]), int(numbers[1])
            correct = length * width
            return [
                f"{correct} square cm",
                f"{length + width} square cm",
                f"{2 * (length + width)} square cm",  # Perimeter
                f"{length * width + 10} square cm"
            ]
    
    elif 'square root' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if numbers:
            num = int(numbers[0])
            import math
            correct = int(math.sqrt(num))
            return [
                str(correct),
                str(correct * 2),
                str(correct + 5),
                str(num // 2)
            ]
    
    # Percentage (e.g. "15% of 200", "Calculate 15% of 200")
    elif '%' in question_text or 'percent' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if len(numbers) >= 2:
            p, n = int(numbers[0]), int(numbers[1])
            correct = round(p * n / 100)
            return [
                str(correct),
                str(correct + 10),
                str(correct - 5) if correct > 5 else str(correct + 5),
                str(n // 2)
            ]
        if len(numbers) == 1:
            p, n = int(numbers[0]), 100
            correct = round(p * n / 100)
            return [str(correct), str(correct + 10), str(correct - 5) if correct > 5 else "5", str(correct + 15)]
    
    # Division (e.g. "72 divided by 8", "What is 72/8")
    elif 'divided' in question_lower or ('/' in question_text and re.search(r'\d+\s*/\s*\d+', question_text)):
        numbers = re.findall(r'\d+', question_text)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            if b != 0:
                correct = a // b
                return [
                    str(correct),
                    str(correct + 1),
                    str(correct - 1) if correct > 0 else str(correct + 2),
                    str(a - b)
                ]
    
    # Volume of cube (e.g. "volume of a cube with side 5cm")
    elif 'volume' in question_lower and 'cube' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if numbers:
            side = int(numbers[0])
            correct = side ** 3
            return [
                str(correct),
                str(side * 6),  # surface area
                str(side * side),
                str(correct + 25)
            ]
    
    # Perimeter of square
    elif 'perimeter' in question_lower and 'square' in question_lower:
        numbers = re.findall(r'\d+', question_text)
        if numbers:
            side = int(numbers[0])
            correct = 4 * side
            return [
                str(correct),
                str(side * side),
                str(side * 2),
                str(correct + 4)
            ]
    
    # Default math options
    return [
        "25",
        "20",
        "30",
        "15"
    ]


def _generate_programming_options(question_text: str) -> List[str]:
    """Generate programming MCQ options using LLM"""
    
    prompt = f"""
Generate exactly 4 realistic MCQ options for this programming question:
Question: {question_text}

Requirements:
1. Return ONLY a JSON array of exactly 4 strings
2. Each option must be a plausible answer candidate
3. Only one option should be clearly correct
4. Other 3 options must be realistic distractors
5. Options must be relevant to programming concepts
6. Do NOT include placeholder text like "Option A" or generic phrases
7. Each option should be a complete, standalone answer

Example format:
["First correct programming option", "Realistic programming distractor 1", "Realistic programming distractor 2", "Realistic programming distractor 3"]

Return ONLY the JSON array, no explanations.
"""
    
    try:
        # For now, return improved static options to avoid async issues
        # In production, this would use LLM call
        question_lower = question_text.lower()
        
        if 'variable' in question_lower and 'what is' in question_lower:
            return [
                "A named storage location in memory that holds a value",
                "A type of loop that repeats code execution",
                "A function that returns a value",
                "A conditional statement for decision making"
            ]
        
        elif 'function' in question_lower:
            return [
                "A reusable block of code that performs a specific task",
                "A variable that stores multiple values",
                "A loop construct for iteration",
                "A data type for text storage"
            ]
        
        elif 'loop' in question_lower:
            return [
                "A control structure that repeats code execution",
                "A way to store data permanently",
                "A method for error handling",
                "A type of variable declaration"
            ]
        
        elif 'array' in question_lower:
            return [
                "A data structure that stores multiple values of the same type",
                "A single value storage location",
                "A function that performs calculations",
                "A conditional statement"
            ]
        
        elif 'inheritance' in question_lower:
            return [
                "A mechanism where a class acquires properties from another class",
                "A method to hide implementation details",
                "A way to create multiple instances of a class",
                "A technique to overload operators"
            ]
        
        elif 'polymorphism' in question_lower:
            return [
                "The ability of different objects to respond to the same message in different ways",
                "Creating a single class with multiple purposes",
                "Hiding data from external access",
                "Combining multiple classes into one"
            ]
        
        elif 'encapsulation' in question_lower:
            return [
                "Bundling data and methods that operate on the data within one unit",
                "Creating multiple classes with similar properties",
                "Converting one data type to another",
                "Reusing code through inheritance"
            ]
        
        elif 'class' in question_lower:
            return [
                "A blueprint for creating objects that defines their properties and behaviors",
                "An instance of an object",
                "A method within an object",
                "A variable that stores data"
            ]
        
        # Fallback to basic options
        return [
            "A named storage location in memory that holds a value",
            "A type of loop that repeats code execution",
            "A function that returns a value",
            "A conditional statement for decision making"
        ]
        
    except Exception as e:
        # Fallback options if any error occurs
        return [
            "A named storage location in memory that holds a value",
            "A type of loop that repeats code execution",
            "A function that returns a value",
            "A conditional statement for decision making"
        ]


def _generate_database_options(question_text: str) -> List[str]:
    """Generate real database options"""
    question_lower = question_text.lower()
    
    if 'primary key' in question_lower:
        return [
            "A unique identifier for each record in a table",
            "A foreign key reference to another table",
            "A data type for text values",
            "A constraint that allows duplicate values"
        ]
    
    elif 'sql' in question_lower and 'what is' in question_lower:
        return [
            "A language for managing and querying relational databases",
            "A programming language for web development",
            "A markup language for document structure",
            "A styling language for web pages"
        ]
    
    elif 'join' in question_lower:
        return [
            "An operation to combine rows from multiple tables",
            "A method to delete records from a table",
            "A way to create a new table",
            "A function to calculate aggregates"
        ]
    
    elif 'normalization' in question_lower:
        return [
            "The process of organizing data to reduce redundancy",
            "A method to encrypt sensitive data",
            "A way to backup database files",
            "A technique to improve query performance"
        ]
    
    return [
        "SELECT * FROM users",
        "INSERT INTO users VALUES",
        "UPDATE users SET",
        "DELETE FROM users WHERE"
    ]


def _generate_web_options(question_text: str) -> List[str]:
    """Generate real web development options"""
    question_lower = question_text.lower()
    
    if 'html' in question_lower and 'what is' in question_lower:
        return [
            "A markup language for creating web page structure",
            "A styling language for web page appearance",
            "A programming language for web interactivity",
            "A database management system"
        ]
    
    elif 'css' in question_lower:
        return [
            "A stylesheet language for describing web page presentation",
            "A programming language for server-side logic",
            "A markup language for document structure",
            "A database query language"
        ]
    
    elif 'javascript' in question_lower:
        return [
            "A programming language for web page interactivity",
            "A markup language for web structure",
            "A styling language for web appearance",
            "A database storage system"
        ]
    
    elif 'responsive design' in question_lower:
        return [
            "Design that adapts to different screen sizes and devices",
            "Design that only works on desktop computers",
            "Design that uses fixed pixel measurements",
            "Design that requires separate mobile sites"
        ]
    
    return [
        "<div>", "<span>", "<p>", "<a>"
    ]


def _generate_oop_options(question_text: str) -> List[str]:
    """Generate real object-oriented programming options"""
    question_lower = question_text.lower()
    
    if 'inheritance' in question_lower:
        return [
            "A mechanism where a class acquires properties from another class",
            "A method to hide implementation details",
            "A way to create multiple instances of a class",
            "A technique to overload operators"
        ]
    
    elif 'encapsulation' in question_lower:
        return [
            "Bundling data and methods that operate on the data within one unit",
            "Creating multiple classes with similar properties",
            "Converting one data type to another",
            "Reusing code through inheritance"
        ]
    
    elif 'polymorphism' in question_lower:
        return [
            "The ability of different objects to respond to the same message in different ways",
            "Creating a single class with multiple purposes",
            "Hiding data from external access",
            "Combining multiple classes into one"
        ]
    
    elif 'class' in question_lower and 'what is' in question_lower:
        return [
            "A blueprint for creating objects that defines their properties and behaviors",
            "An instance of an object",
            "A method within an object",
            "A variable that stores data"
        ]
    
    return [
        "public", "private", "protected", "static"
    ]


def _generate_general_options(question_text: str) -> List[str]:
    """Generate real options for general questions"""
    question_lower = question_text.lower()
    
    if 'define' in question_lower or 'what is' in question_lower:
        return [
            "A clear explanation of the concept or term",
            "An example of the concept in practice",
            "A comparison with related concepts",
            "A historical background of the concept"
        ]
    
    elif 'explain' in question_lower:
        return [
            "A detailed description of how something works",
            "A brief definition of the term",
            "An example of usage",
            "A comparison with alternatives"
        ]
    
    elif 'compare' in question_lower:
        return [
            "An analysis of similarities and differences",
            "A description of only similarities",
            "A description of only differences",
            "A historical overview"
        ]
    
    return [
        "The first option is correct",
        "The second option is correct", 
        "The third option is correct",
        "The fourth option is correct"
    ]


def _generate_mcq_options_batch(questions: List[str]) -> Dict[str, Dict[str, Any]]:
    """Generate complete MCQ structure for all questions in a single LLM call"""
    import json
    
    if not questions:
        return {}
    
    # Format questions for the prompt
    questions_list = "\n".join([f"- {q}" for q in questions])
    
    prompt = f"""You are generating high-quality multiple-choice questions for an assessment.

Generate complete MCQs for each question below.

For each question, provide:
- Exactly 4 options
- 1 correct answer
- 3 realistic distractors
- Difficulty classification (easy/medium/hard)
- Marks based on difficulty (easy=1, medium=2, hard=3)

Requirements (apply to ALL topics — math, programming, general knowledge, etc.):
* correct_answer must be the ACTUAL correct answer to the question. Do not guess; for calculations/computations you MUST compute the real result.
* Options must be contextually relevant to the question (e.g. numeric options for arithmetic, programming concepts for code questions, definitions for "what is X").
* "correct_answer" must exactly match one of the options (same string).
* Options must be unique and meaningful. Shuffle option order so the correct answer is not always first.
* Avoid generic placeholders like "public, private, protected, static", "compile the code, run the debugger", "none of the above", "all of the above".

For numeric/calculation questions (arithmetic, percentage, division, square root, area, volume, etc.):
* Compute the exact answer and use it as correct_answer. Provide 3 other plausible numeric distractors.

Difficulty Classification:
* Easy – basic definitions, recall questions
* Medium – explanation-based, conceptual understanding
* Hard – scenario-based, analytical, deeper reasoning

Output format (JSON list only, no explanations):
[
  {{
    "type": "mcq",
    "question": "What is a variable in programming?",
    "options": [
      "A named storage location used to hold data",
      "A function used to repeat code",
      "A compiler instruction",
      "A programming language keyword"
    ],
    "correct_answer": "A named storage location used to hold data",
    "difficulty": "easy",
    "marks": 1
  }}
]

Questions:
{questions_list}"""
    
    try:
        llm_response = asyncio.run(invoke_llm(
            prompt=prompt,
            task_type=TaskType.ASSESSMENT_GENERATION,
            agent_name="assessment_builder_agent_batch_mcq",
            response_mime_type="application/json",
            max_output_tokens=8192,
        ))
        
        # Parse LLM response
        response_text = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
        parsed = _parse_llm_json(response_text)
        
        # Validate and clean the response
        mcq_map = {}
        if isinstance(parsed, list):
            for mcq_item in parsed:
                if isinstance(mcq_item, dict):
                    question = mcq_item.get('question', '')
                    if question and _validate_mcq_structure(mcq_item):
                        mcq_map[question] = mcq_item
        
        return mcq_map
        
    except Exception as e:
        # Fallback: return empty dict to trigger individual generation
        return {}


def _validate_mcq_structure(mcq_item: Dict[str, Any]) -> bool:
    """Validate MCQ structure meets all requirements"""
    try:
        # Check required fields
        required_fields = ['type', 'question', 'options', 'correct_answer', 'difficulty', 'marks']
        for field in required_fields:
            if field not in mcq_item:
                return False
        
        # Check type
        if mcq_item.get('type') != 'mcq':
            return False
        
        # Check options
        options = mcq_item.get('options', [])
        if not isinstance(options, list) or len(options) != 4:
            return False
        
        # Check all options are strings and non-empty
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                return False
        
        # Check correct answer exists in options
        correct_answer = mcq_item.get('correct_answer', '')
        if not correct_answer or correct_answer not in options:
            return False
        
        # Check difficulty
        difficulty = mcq_item.get('difficulty', '')
        if difficulty not in ['easy', 'medium', 'hard']:
            return False
        
        # Check marks
        marks = mcq_item.get('marks', 0)
        expected_marks = {'easy': 1, 'medium': 2, 'hard': 3}
        if marks != expected_marks.get(difficulty, 0):
            return False
        
        # Check for placeholder text
        placeholder_phrases = [
            'public', 'private', 'protected', 'static',
            'compile the code', 'run the debugger', 'check syntax errors', 'add comments',
            'none of the above', 'all of the above'
        ]
        
        for opt in options:
            opt_lower = opt.lower()
            if any(phrase in opt_lower for phrase in placeholder_phrases):
                return False
        
        return True
        
    except Exception:
        return False


def _compute_expected_math_answer(question_text: str) -> Optional[str]:
    """
    Compute expected numeric answer for math-like questions.
    Returns the expected answer string, or None if not a recognized math pattern.
    Used to validate batch LLM output for calculation questions.
    """
    question_lower = question_text.lower()
    numbers = re.findall(r'\d+', question_text)
    if not numbers:
        return None
    try:
        if '+' in question_text and 'what is' in question_lower and len(numbers) >= 2:
            return str(int(numbers[0]) + int(numbers[1]))
        if ('×' in question_text or 'multiply' in question_lower) and len(numbers) >= 2:
            return str(int(numbers[0]) * int(numbers[1]))
        if 'area' in question_lower and 'rectangle' in question_lower and len(numbers) >= 2:
            return f"{int(numbers[0]) * int(numbers[1])} square cm"
        if 'square root' in question_lower and numbers:
            import math
            return str(int(math.sqrt(int(numbers[0]))))
        if '%' in question_text or 'percent' in question_lower:
            if len(numbers) >= 2:
                return str(round(int(numbers[0]) * int(numbers[1]) / 100))
            if len(numbers) == 1:
                return str(round(int(numbers[0])))
        if 'divided' in question_lower or re.search(r'\d+\s*/\s*\d+', question_text):
            if len(numbers) >= 2 and int(numbers[1]) != 0:
                return str(int(numbers[0]) // int(numbers[1]))
        if 'volume' in question_lower and 'cube' in question_lower and numbers:
            s = int(numbers[0])
            return str(s ** 3)
        if 'perimeter' in question_lower and 'square' in question_lower and numbers:
            return str(4 * int(numbers[0]))
        return None
    except (ValueError, ZeroDivisionError):
        return None


def _create_assessment_from_auto_processing(
    raw_text: str, 
    organization_type: str, 
    detected_sections: List[str], 
    extracted_questions: List[Dict[str, Any]],
    total_marks_input: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create structured assessment from auto-processed content.
    
    Args:
        raw_text: Original raw assessment text
        organization_type: 'corporate' or 'university'
        detected_sections: List of detected section names
        extracted_questions: List of extracted questions
        
    Returns:
        Structured assessment dictionary
    """
    # Extract title from raw text
    title = "Auto-Generated Assessment"
    title_match = re.search(r'(?i)(?:title|assessment|exam|test)\s*[:\-]?\s*([^\n]+)', raw_text)
    if title_match:
        title = title_match.group(1).strip()
    
    # Create sections with questions
    # Collect all question texts for batch generation
    all_questions = [q.get('question', '') for q in extracted_questions if q.get('question')]
    
    # Generate complete MCQs for all questions in one LLM call
    mcq_map = _generate_mcq_options_batch(all_questions)
    
    sections = []
    questions_per_section = max(1, len(extracted_questions) // len(detected_sections))
    
    for i, section_name in enumerate(detected_sections):
        # Get questions for this section
        start_idx = i * questions_per_section
        end_idx = start_idx + questions_per_section if i < len(detected_sections) - 1 else len(extracted_questions)
        section_questions = extracted_questions[start_idx:end_idx]
        
        # Convert each question to MCQ format using batch-generated complete structure
        mcq_questions = []
        for j, q_data in enumerate(section_questions):
            question_text = q_data.get('question', '')
            
            # Get complete MCQ structure from batch generation
            batch_mcq = mcq_map.get(question_text, {})
            
            # For math-like questions, reject batch if correct_answer doesn't match computed value
            if batch_mcq and _validate_mcq_structure(batch_mcq):
                expected_math = _compute_expected_math_answer(question_text)
                if expected_math is not None:
                    if str(batch_mcq.get("correct_answer", "")).strip() != str(expected_math).strip():
                        batch_mcq = {}
            
            if batch_mcq and _validate_mcq_structure(batch_mcq):
                # Use batch-generated complete MCQ structure
                mcq_question = {
                    "type": "mcq",
                    "question": question_text,
                    "options": batch_mcq["options"],
                    "correct_answer": batch_mcq["correct_answer"],
                    "difficulty": batch_mcq["difficulty"],
                    "marks": batch_mcq["marks"]
                }
            else:
                # Fallback: rule-based generators put correct answer first; use it then shuffle
                difficulty = _infer_difficulty(question_text)
                options = _generate_mcq_options(question_text, difficulty)
                correct_answer = options[0] if options else ""
                random.shuffle(options)
                
                # Assign marks by difficulty (easy=1, medium=2, hard=3)
                marks = MARKS_BY_DIFFICULTY.get(difficulty, 2)
                
                mcq_question = {
                    "type": "mcq",
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "difficulty": difficulty,
                    "marks": marks
                }
            
            mcq_questions.append(mcq_question)
        
        if mcq_questions:
            sections.append({
                "section_name": section_name,
                "weight": 0,  # Will be calculated later
                "questions": mcq_questions
            })
    
    # Distribute marks and calculate weights
    if sections:
        # Use user-provided total_marks if available, otherwise use TARGET_TOTAL_MARKS
        target_total = total_marks_input if total_marks_input is not None else TARGET_TOTAL_MARKS
        sections, total_marks = _distribute_marks_and_weights(sections, target_total)
    else:
        # Create a default section if no questions found
        target_total = total_marks_input if total_marks_input is not None else TARGET_TOTAL_MARKS
        sections = [{
            "section_name": "General Assessment",
            "weight": 100,
            "questions": [{
                "type": "mcq",
                "question": "General assessment question based on provided content",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "difficulty": "medium",
                "marks": target_total
            }]
        }]
        total_marks = target_total
    
    # Create final assessment structure
    assessment = {
        "title": title,
        "organization_type": organization_type,
        "sections": sections,
        "total_marks": total_marks,
        "metadata": {
            "generated_by": "assessment_builder_agent",
            "generated_at": datetime.utcnow().isoformat(),
            "auto_processed": True,
            "original_sections_detected": len(detected_sections),
            "questions_extracted": len(extracted_questions)
        }
    }
    
    return assessment


def _parse_llm_json(llm_response_text: str) -> Dict[str, Any]:
    """
    Robust JSON parsing helper for LLM output.
    
    Args:
        llm_response_text: Raw text response from LLM
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed or is invalid
    """
    if not llm_response_text or not isinstance(llm_response_text, str):
        raise ValueError("LLM response text is empty or invalid")
    
    text = llm_response_text.strip()
    
    # Remove markdown code blocks
    if "```json" in text:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    
    # Extract first JSON object if multiple exist
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in LLM response")
    
    # Find balanced JSON object
    depth = 0
    in_string = False
    escape_char = False
    json_end = -1
    
    for i in range(json_start, len(text)):
        char = text[i]
        
        if in_string:
            if escape_char:
                escape_char = False
            elif char == "\\":
                escape_char = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
    
    if json_end == -1:
        raise ValueError("Incomplete JSON object in LLM response")
    
    json_text = text[json_start:json_end]
    
    # Replace single quotes with double quotes (common LLM issue)
    json_text = re.sub(r"'([^']*)'\":", r'"\1":', json_text)
    json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)
    
    # Remove trailing commas
    json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)
    
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("JSON is not an object")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


SYSTEM_INSTRUCTION = """
You are an expert assessment designer.
Your task is to convert messy, human-written assessment descriptions into a clean, multiple-choice assessment JSON.

STRICT OUTPUT REQUIREMENTS:
- Output ONLY valid JSON. No markdown, no comments, no code fences, and no explanations.
- The JSON MUST match exactly this schema (field names and types):
{
  "title": "string",
  "organization_type": "corporate or university",
  "sections": [
    {
      "section_name": "string",
      "weight": 0,
      "questions": [
        {
          "type": "mcq",
          "question": "string",
          "options": ["string", "string", "string", "string"],
          "correct_answer": "string",
          "difficulty": "easy or medium or hard",
          "marks": 0
        }
      ]
    }
  ],
  "total_marks": 0,
  "metadata": {
    "generated_by": "assessment_builder_agent",
    "generated_at": "ISO 8601 timestamp string"
  }
}

TRANSFORMATION RULES:
- Extract a concise, informative assessment title.
- Identify logical sections or topics from the text and map them into `sections`.
- Convert every question into a 4-option MCQ:
  - If the original question is not an MCQ, create 4 plausible answer options.
  - Ensure EXACTLY 4 options per question.
  - Ensure there is a single correct answer.
  - Set `correct_answer` to the exact text of the correct option.
- Infer difficulty (`easy`, `medium`, `hard`) from wording and cognitive complexity.
- Infer correct answers when they are not explicitly specified.
- Ignore any scores, marks, or weights mentioned in the text. Focus only on question content and structure.
- Leave `weight`, `marks`, and `total_marks` as integers with placeholder value 0. The system will recalculate these.
- Keep language clear, professional, and unambiguous.

ORGANIZATION-SPECIFIC GUIDANCE:
- For corporate organizations:
  - Focus on practical skill evaluation and real-world scenario MCQs.
  - Aim for a balanced difficulty mix (about 30% easy, 50% medium, 20% hard).
- For university organizations:
  - Use an academic structure with clear section-based segmentation (units, modules, topics).
  - Include a mix of theoretical and applied questions.

Return ONLY the JSON object. Do not wrap it in backticks. Do not include any additional text.
"""


def _normalize_correct_answer(raw_correct: Any, options: List[str]) -> str:
    """Normalize correct_answer to be one of the option texts."""
    if not options:
        return ""

    if raw_correct is None:
        return options[0]

    s = str(raw_correct).strip()
    if not s:
        return options[0]

    # Exact text match (case-insensitive)
    for opt in options:
        if s.lower() == opt.strip().lower():
            return opt

    # Letter-based match
    letters = ["A", "B", "C", "D"]
    upper = s.upper()
    if upper in letters:
        idx = letters.index(upper)
        if idx < len(options):
            return options[idx]

    # 1-based index match
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(options):
            return options[idx]

    return options[0]


def _normalize_question(raw_q: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Coerce a raw question dict into the target MCQ schema."""
    question_text = str(
        raw_q.get("question")
        or raw_q.get("text")
        or raw_q.get("prompt")
        or ""
    ).strip()
    if not question_text:
        return None

    raw_options = raw_q.get("options") or raw_q.get("choices") or []
    options: List[str] = []
    if isinstance(raw_options, list):
        for opt in raw_options:
            if isinstance(opt, str):
                text = opt.strip()
            elif isinstance(opt, dict):
                text = str(
                    opt.get("text")
                    or opt.get("label")
                    or opt.get("option")
                    or ""
                ).strip()
            else:
                text = ""
            if text:
                options.append(text)

    letters = ["A", "B", "C", "D"]
    # Ensure exactly 4 options
    options = options[:4]
    while len(options) < 4:
        options.append(f"Option {letters[len(options)]}")

    correct_raw = raw_q.get("correct_answer") or raw_q.get("answer")
    correct_answer = _normalize_correct_answer(correct_raw, options)

    difficulty = str(raw_q.get("difficulty") or "medium").lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    return {
        "type": "mcq",
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "difficulty": difficulty,
        "marks": 0,
    }


MARKS_BY_DIFFICULTY = {"easy": 1, "medium": 2, "hard": 3}


def _distribute_marks_and_weights(
    sections: List[Dict[str, Any]],
    total_marks_target: int,
) -> (List[Dict[str, Any]], int):
    """Assign marks by difficulty (easy=1, medium=2, hard=3); compute section weights from those marks."""
    total_questions = sum(len(sec.get("questions") or []) for sec in sections)
    if total_questions == 0:
        raise ValueError("No questions found to allocate marks.")

    # Allot marks by difficulty: easy=1, medium=2, hard=3 (do not overwrite if already 1/2/3)
    for sec in sections:
        for q in sec.get("questions", []):
            current = q.get("marks", 0)
            if current not in (1, 2, 3):
                difficulty = str(q.get("difficulty") or "medium").lower()
                if difficulty not in ("easy", "medium", "hard"):
                    difficulty = "medium"
                q["marks"] = MARKS_BY_DIFFICULTY.get(difficulty, 2)

    # Total marks = sum of question marks (no fixed target; total follows difficulty mix)
    total_marks = sum(
        q.get("marks", 0)
        for sec in sections
        for q in sec.get("questions", [])
    )
    if total_marks <= 0:
        total_marks = total_marks_target

    # Section weights as percentage of total marks, normalized to 100
    remaining_weight = 100
    for idx, sec in enumerate(sections):
        sec_marks = sum(q.get("marks", 0) for q in sec.get("questions", []))
        if total_marks <= 0:
            weight = 0
        elif idx == len(sections) - 1:
            # Last section takes remaining to ensure sum is exactly 100
            weight = remaining_weight
        else:
            weight = round((sec_marks / total_marks) * 100)
            remaining_weight -= weight
        sec["weight"] = max(0, int(weight))

    return sections, total_marks


def _normalize_assessment(
    raw: Dict[str, Any],
    organization_type: str,
    target_total: int = TARGET_TOTAL_MARKS,
) -> Dict[str, Any]:
    """Normalize raw LLM JSON into the strict assessment schema and auto-calc scores."""
    title = str(raw.get("title") or "Untitled Assessment").strip()
    sections_raw = raw.get("sections") or []

    if not isinstance(sections_raw, list):
        raise ValueError("sections must be a list in LLM output.")

    normalized_sections: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections_raw, start=1):
        if not isinstance(sec, dict):
            continue

        section_name = str(
            sec.get("section_name")
            or sec.get("name")
            or sec.get("title")
            or f"Section {idx}"
        ).strip()

        questions_raw = sec.get("questions") or []
        if not isinstance(questions_raw, list):
            continue

        normalized_questions: List[Dict[str, Any]] = []
        for q_idx, q in enumerate(questions_raw, start=1):
            if not isinstance(q, dict):
                continue
            q_norm = _normalize_question(q, q_idx)
            if q_norm:
                normalized_questions.append(q_norm)

        if normalized_questions:
            normalized_sections.append(
                {
                    "section_name": section_name,
                    "weight": 0,
                    "questions": normalized_questions,
                }
            )

    if not normalized_sections:
        raise ValueError("No valid sections/questions found in LLM output.")

    normalized_sections, total_marks = _distribute_marks_and_weights(
        normalized_sections, target_total
    )

    assessment: Dict[str, Any] = {
        "title": title,
        "organization_type": organization_type,
        "sections": normalized_sections,
        "total_marks": total_marks,
        "metadata": {
            "generated_by": "assessment_builder_agent",
            "generated_at": datetime.utcnow().isoformat(),
        },
    }
    return assessment


def _build_markdown_preview(assessment: Dict[str, Any]) -> str:
    """Generate beautified Markdown preview for the structured assessment."""
    title = assessment.get("title") or "Assessment"
    org_type = assessment.get("organization_type") or ""
    org_label = org_type.capitalize() if isinstance(org_type, str) else str(org_type)
    total_marks = assessment.get("total_marks") or TARGET_TOTAL_MARKS

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Organization Type:** {org_label}  ")
    lines.append(f"**Total Marks:** {total_marks}  ")
    lines.append("")
    lines.append("---")

    sections = assessment.get("sections") or []
    letters = ["A", "B", "C", "D"]

    for s_idx, sec in enumerate(sections, start=1):
        section_name = sec.get("section_name") or f"Section {s_idx}"
        weight = sec.get("weight", 0)
        lines.append("")
        lines.append(f"## Section {s_idx}: {section_name} (Weight: {weight})")
        lines.append("")

        questions = sec.get("questions") or []
        for q_idx, q in enumerate(questions, start=1):
            q_text = str(q.get("question") or "").strip()
            options = q.get("options") or []
            marks = q.get("marks", 0)
            difficulty = str(q.get("difficulty") or "").capitalize()
            correct = str(q.get("correct_answer") or "").strip()

            lines.append(f"{q_idx}. {q_text}  ")
            for o_idx, opt in enumerate(options[:4]):
                label = letters[o_idx] if o_idx < len(letters) else chr(ord("A") + o_idx)
                lines.append(f"{label}. {opt}  ")

            # Derive correct answer letter if possible
            correct_letter = ""
            for o_idx, opt in enumerate(options[:4]):
                if correct and opt.strip().lower() == correct.lower():
                    correct_letter = letters[o_idx]
                    break
            if not correct_letter and correct:
                upper = correct.upper()
                if upper in letters:
                    correct_letter = upper
                else:
                    correct_letter = "A" if options else ""

            lines.append("")
            lines.append(f"   Correct Answer: {correct_letter}  ")
            lines.append(f"   Marks: {marks}  ")
            lines.append(f"   Difficulty: {difficulty}  ")
            lines.append("")

    return "\n".join(lines).strip()


@traceable(name="assessment_builder_agent")
async def assessment_builder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert messy manual assessment text into a structured assessment and Markdown preview.

    Expected input in state:
      - manual_assessment_text (or body.raw_assessment_text)
      - organization_type ("corporate" | "university")
    """
    tenant_id = (
        state.get("tenant_id")
        or state.get("body", {}).get("tenant_id")
        or "default_tenant"
    )
    log_context = create_log_context("assessment_builder_agent", tenant_id)
    start_time = log_context["start_time"]

    body = state.get("body", {}) or {}

    # Defensive: ensure callback_url is present (entrypoint already enforces this for /analyze-resume-callback)
    callback_url = (
        state.get("callback_url")
        or body.get("callback_url")
        or ""
    )
    if not callback_url:
        processing_time = _calculate_processing_time(start_time)
        error_msg = "Missing required field: callback_url"
        log.warning(f"ASSESSMENT_BUILDER: {error_msg}")
        error_state = {
            "ok": False,
            "error": error_msg,
            "validation_error": error_msg,
            "analysis_type": "manual_assessment_builder",
            "processing_time_seconds": processing_time,
        }
        log_agent_completion(
            log_context,
            {"success": False, "error": error_msg},
            "deterministic",
            processing_time,
        )
        return error_state

    # Extract total_marks from input payload
    total_marks_input = (
        state.get("total_marks") or
        body.get("total_marks")
    )
    
    raw_text = (
        state.get("manual_assessment_text")
        or body.get("raw_assessment_text")
        or body.get("rawAssessmentText")
        or ""
    )

    org_type_raw = (
        state.get("organization_type")
        or body.get("organization_type")
        or body.get("organizationType")
        or ""
    )
    org_type_lower = str(org_type_raw).strip().lower()
    if org_type_lower.startswith("corp"):
        organization_type = "corporate"
    elif org_type_lower.startswith("univ"):
        organization_type = "university"
    elif org_type_lower in {"corporate", "university"}:
        organization_type = org_type_lower
    else:
        organization_type = ""

    if not raw_text or not str(raw_text).strip():
        processing_time = _calculate_processing_time(start_time)
        error_msg = "Missing required field: raw_assessment_text"
        log.warning(f"ASSESSMENT_BUILDER: {error_msg}")
        error_state = {
            "ok": False,
            "error": error_msg,
            "validation_error": error_msg,
            "analysis_type": "manual_assessment_builder",
            "processing_time_seconds": processing_time,
        }
        log_agent_completion(
            log_context,
            {"success": False, "error": error_msg},
            "deterministic",
            processing_time,
        )
        return error_state

    if organization_type not in {"corporate", "university"}:
        processing_time = _calculate_processing_time(start_time)
        error_msg = (
            "Invalid or missing organization_type (expected 'corporate' or 'university')"
        )
        log.warning(f"ASSESSMENT_BUILDER: {error_msg}")
        error_state = {
            "ok": False,
            "error": error_msg,
            "validation_error": error_msg,
            "analysis_type": "manual_assessment_builder",
            "processing_time_seconds": processing_time,
        }
        log_agent_completion(
            log_context,
            {"success": False, "error": error_msg},
            "deterministic",
            processing_time,
        )
        return error_state

    # Clamp very long inputs for safety
    raw_text_str = str(raw_text)
    if len(raw_text_str) > MAX_INPUT_LENGTH:
        raw_text_str = raw_text_str[:MAX_INPUT_LENGTH]

    prompt = (
        f'ORGANIZATION_TYPE: "{organization_type}"\n'
        f"RAW_ASSESSMENT_TEXT:\n"
        f"\"\"\"{raw_text_str}\"\"\""
    )

    try:
        llm_response = await asyncio.wait_for(
            invoke_llm(
                prompt=prompt,
                task_type=TaskType.ASSESSMENT_GENERATION,
                agent_name="assessment_builder_agent",
                response_mime_type="application/json",
                max_output_tokens=2048,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        processing_time = _calculate_processing_time(start_time)
        error_msg = "LLM timed out while generating assessment."
        log.error(f"ASSESSMENT_BUILDER: {error_msg}")
        error_state = {
            "ok": False,
            "error": error_msg,
            "analysis_type": "manual_assessment_builder",
            "processing_time_seconds": processing_time,
        }
        log_agent_completion(
            log_context,
            {"success": False, "error": error_msg},
            "llm",
            processing_time,
        )
        return error_state
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        error_msg = f"LLM invocation failed: {e}"
        log.error(f"ASSESSMENT_BUILDER: {error_msg}")
        error_state = {
            "ok": False,
            "error": error_msg,
            "analysis_type": "manual_assessment_builder",
            "processing_time_seconds": processing_time,
        }
        log_agent_completion(
            log_context,
            {"success": False, "error": error_msg},
            "llm",
            processing_time,
        )
        return error_state

    try:
        # Parse LLM response using robust helper
        response_text = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
        parsed = _parse_llm_json(response_text)
        
        # Normalize and structure the assessment
        # Use user-provided total_marks if available, otherwise use TARGET_TOTAL_MARKS
        target_total = total_marks_input if total_marks_input is not None else TARGET_TOTAL_MARKS
        structured_assessment = _normalize_assessment(parsed, organization_type, target_total)
        
        # Calculate total marks from structured assessment
        total_marks = structured_assessment.get("total_marks", 0)
        
        # Generate markdown preview
        preview_markdown = _build_markdown_preview(structured_assessment)
        
        # Update state with successful results
        result_state = {
            "ok": True,
            "analysis_type": "manual_assessment_builder",
            "manual_assessment_text": raw_text_str,
            "organization_type": organization_type,
            "structured_manual_assessment": structured_assessment,
            "beautified_preview": preview_markdown,
            # Fields used by callback/response layer
            "structured_assessment": structured_assessment,
            "preview_markdown": preview_markdown,
            "total_marks": total_marks,
            "processing_time_seconds": _calculate_processing_time(start_time),
        }
        
        log_agent_completion(
            log_context,
            {"success": True, "confidence_score": 0.9},
            "llm",
            result_state["processing_time_seconds"],
        )
        
        return result_state
        
    except Exception as e:
        # If LLM parsing fails, try auto-processing the raw text
        try:
            log.info(f"ASSESSMENT_BUILDER: LLM parsing failed, attempting auto-processing: {e}")
            
            # Auto-detect sections from raw text
            detected_sections = _detect_sections_from_text(raw_text_str)
            
            # Extract questions from raw text
            extracted_questions = _extract_questions_from_text(raw_text_str)
            
            # Create structured assessment from auto-processed content
            structured_assessment = _create_assessment_from_auto_processing(
                raw_text_str, organization_type, detected_sections, extracted_questions, total_marks_input
            )
            
            # Calculate total marks
            total_marks = structured_assessment.get("total_marks", 0)
            
            # Generate markdown preview
            preview_markdown = _build_markdown_preview(structured_assessment)
            
            # Update state with successful auto-processing results
            result_state = {
                "ok": True,
                "analysis_type": "manual_assessment_builder",
                "manual_assessment_text": raw_text_str,
                "organization_type": organization_type,
                "structured_manual_assessment": structured_assessment,
                "beautified_preview": preview_markdown,
                # Fields used by callback/response layer
                "structured_assessment": structured_assessment,
                "preview_markdown": preview_markdown,
                "total_marks": total_marks,
                "processing_time_seconds": _calculate_processing_time(start_time),
                "auto_processed": True,  # Flag to indicate auto-processing was used
            }
            
            log_agent_completion(
                log_context,
                {"success": True, "confidence_score": 0.7, "method": "auto_processing"},
                "deterministic",
                result_state["processing_time_seconds"],
            )
            
            return result_state
            
        except Exception as auto_error:
            processing_time = _calculate_processing_time(start_time)
            error_msg = f"Failed to parse or normalize LLM output and auto-processing failed: {e} | Auto-error: {auto_error}"
            log.error(f"ASSESSMENT_BUILDER: {error_msg}")
            error_state = {
                "ok": False,
                "error": error_msg,
                "analysis_type": "manual_assessment_builder",
                "processing_time_seconds": processing_time,
            }
            log_agent_completion(
                log_context,
                {"success": False, "error": error_msg},
                "llm",
                processing_time,
            )
            return error_state


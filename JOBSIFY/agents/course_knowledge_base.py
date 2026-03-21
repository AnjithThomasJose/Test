"""
Course Knowledge Base using ChromaDB
Stores and retrieves course recommendations using vector search
"""

import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from chroma import client, embedding_fn, normalize_metadata, _get_collection

log = logging.getLogger(__name__)

class CourseKnowledgeBase:
    """Manages course data in ChromaDB for semantic search"""
    
    def __init__(self):
        self.client = client
        self.embedding_fn = embedding_fn
        self.collection = _get_collection("courses_knowledge_base")
        log.info("✅ Course Knowledge Base initialized")
    
    def _create_searchable_text(self, course_data: Dict[str, Any]) -> str:
        """Create searchable text from course data for embedding"""
        parts = []
        
        # Title
        if course_data.get("title"):
            parts.append(f"Title: {course_data['title']}")
        
        # Description
        if course_data.get("description"):
            parts.append(f"Description: {course_data['description']}")
        
        # Skills
        if course_data.get("skills"):
            skills = course_data["skills"]
            if isinstance(skills, list):
                skills_str = ", ".join(skills)
            else:
                skills_str = str(skills)
            parts.append(f"Skills: {skills_str}")
        
        # Provider
        if course_data.get("provider"):
            parts.append(f"Platform: {course_data['provider']}")
        
        # Author (for books)
        if course_data.get("author"):
            parts.append(f"Author: {course_data['author']}")
        
        return " | ".join(parts)
    
    def _generate_course_id(self, course: Dict[str, Any]) -> str:
        """Generate unique ID from course URL, provider, and title"""
        # Include provider and title to ensure uniqueness even if URLs are similar
        url = course.get("url", "")
        provider = course.get("provider", "")
        title = course.get("title", "")
        
        # Create a unique identifier combining all three
        unique_string = f"{provider}|{url}|{title}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def add_course(self, course_id: str, course_data: Dict[str, Any]) -> bool:
        """
        Add or update a course in the knowledge base.
        
        Args:
            course_id: Unique identifier (e.g., URL hash or platform_id)
            course_data: Course information dict with:
                - title: Course title
                - provider: Platform name (Coursera, Udemy, etc.)
                - url: Course URL
                - description: Course description
                - skills: List of skills covered
                - difficulty: Beginner/Intermediate/Advanced
                - duration: Course duration
                - type: course/book/paper/video
                - author: Author name (for books)
        
        Returns:
            True if successful, False otherwise
        """
        # Create searchable text from course data
        searchable_text = self._create_searchable_text(course_data)
        
        # Prepare metadata (filtered for ChromaDB compatibility)
        skills_value = course_data.get("skills", [])
        if isinstance(skills_value, list):
            skills_str = ", ".join(str(s) for s in skills_value)[:500]
        else:
            skills_str = str(skills_value)[:500]
        
        metadata = normalize_metadata({
            "course_id": course_id,
            "title": course_data.get("title", "")[:200],
            "provider": course_data.get("provider", ""),
            "url": course_data.get("url", ""),
            "difficulty": course_data.get("difficulty", "All Levels"),
            "duration": course_data.get("duration", ""),
            "type": course_data.get("type", "course"),
            "author": course_data.get("author", ""),
            "skills": skills_str,
        })
        
        try:
            self.collection.upsert(
                ids=[course_id],
                documents=[searchable_text],
                metadatas=[metadata]
            )
            log.info(f"✅ Course {course_id} added to knowledge base")
            return True
        except Exception as e:
            log.error(f"❌ Error adding course {course_id}: {e}")
            return False
    
    def batch_add_courses(self, courses: List[Dict[str, Any]]) -> bool:
        """
        Add multiple courses in batch.
        
        Args:
            courses: List of course dictionaries
        
        Returns:
            True if successful, False otherwise
        """
        if not courses:
            return False
        
        ids = []
        documents = []
        metadatas = []
        
        for course in courses:
            course_id = course.get("course_id") or self._generate_course_id(course)
            searchable_text = self._create_searchable_text(course)
            
            skills_value = course.get("skills", [])
            if isinstance(skills_value, list):
                skills_str = ", ".join(str(s) for s in skills_value)[:500]
            else:
                skills_str = str(skills_value)[:500]
            
            metadata = normalize_metadata({
                "course_id": course_id,
                "title": course.get("title", "")[:200],
                "provider": course.get("provider", ""),
                "url": course.get("url", ""),
                "difficulty": course.get("difficulty", "All Levels"),
                "duration": course.get("duration", ""),
                "type": course.get("type", "course"),
                "author": course.get("author", ""),
                "skills": skills_str,
            })
            
            ids.append(course_id)
            documents.append(searchable_text)
            metadatas.append(metadata)
        
        try:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            log.info(f"✅ Added {len(courses)} courses to knowledge base")
            return True
        except Exception as e:
            log.error(f"❌ Error batch adding courses: {e}")
            return False
    
    def search_courses(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search courses using semantic similarity.
        
        Args:
            query: Search query (e.g., "React.js courses for beginners")
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"type": "course", "difficulty": "Beginner"})
        
        Returns:
            List of course dictionaries with similarity scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters
            )
            
            courses = []
            if results.get("ids") and results["ids"][0]:
                for i, course_id in enumerate(results["ids"][0]):
                    metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                    distance = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
                    
                    # Convert skills string back to list
                    skills_str = metadata.get("skills", "")
                    skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
                    
                    # Convert back to course dict
                    course = {
                        "course_id": course_id,
                        "title": metadata.get("title", ""),
                        "provider": metadata.get("provider", ""),
                        "url": metadata.get("url", ""),
                        "difficulty": metadata.get("difficulty", "All Levels"),
                        "duration": metadata.get("duration", ""),
                        "type": metadata.get("type", "course"),
                        "author": metadata.get("author", ""),
                        "similarity_score": 1.0 - distance,  # Convert distance to similarity
                        "skills": skills_list
                    }
                    courses.append(course)
            
            return courses
        except Exception as e:
            log.error(f"❌ Error searching courses: {e}")
            return []
    
    def search_by_skills(
        self, 
        skills: List[str], 
        top_k: int = 10,
        difficulty: Optional[str] = None,
        course_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search courses by skills.
        
        Args:
            skills: List of skills to search for
            top_k: Number of results
            difficulty: Optional difficulty filter
            course_type: Optional type filter (course/book/paper/video)
        
        Returns:
            List of matching courses
        """
        query = f"Courses covering: {', '.join(skills)}"
        filters = {}
        
        if difficulty:
            filters["difficulty"] = difficulty
        if course_type:
            filters["type"] = course_type
        
        return self.search_courses(query, top_k=top_k, filters=filters if filters else None)
    
    def get_course_count(self) -> int:
        """Get total number of courses in knowledge base"""
        try:
            return self.collection.count()
        except Exception as e:
            log.error(f"❌ Error getting course count: {e}")
            return 0
    
    def get_course_by_id(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific course by ID"""
        try:
            results = self.collection.get(ids=[course_id])
            if results.get("ids") and results["ids"]:
                metadata = results.get("metadatas", [[]])[0] if results.get("metadatas") else {}
                document = results.get("documents", [[]])[0] if results.get("documents") else ""
                
                skills_str = metadata.get("skills", "")
                skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
                
                return {
                    "course_id": course_id,
                    "title": metadata.get("title", ""),
                    "provider": metadata.get("provider", ""),
                    "url": metadata.get("url", ""),
                    "difficulty": metadata.get("difficulty", "All Levels"),
                    "duration": metadata.get("duration", ""),
                    "type": metadata.get("type", "course"),
                    "author": metadata.get("author", ""),
                    "skills": skills_list,
                    "description": document  # Full searchable text
                }
            return None
        except Exception as e:
            log.error(f"❌ Error getting course {course_id}: {e}")
            return None
    
    def delete_course_by_url(self, url: str) -> bool:
        """
        Delete a course by URL.
        
        Args:
            url: Course URL to delete
            
        Returns:
            True if course was found and deleted, False otherwise
        """
        try:
            # Get all courses with matching URL
            results = self.collection.get(
                where={"url": url}
            )
            
            if not results.get("ids") or not results["ids"]:
                log.warning(f"⚠️ No course found with URL: {url}")
                return False
            
            # Delete all matching courses (should typically be just one)
            ids_to_delete = results["ids"]
            self.collection.delete(ids=ids_to_delete)
            
            log.info(f"✅ Deleted {len(ids_to_delete)} course(s) with URL: {url}")
            return True
        except Exception as e:
            log.error(f"❌ Error deleting course by URL {url}: {e}")
            return False
    
    def delete_courses_by_urls(self, urls: List[str]) -> int:
        """
        Delete multiple courses by URLs.
        
        Args:
            urls: List of course URLs to delete
            
        Returns:
            Number of courses deleted
        """
        deleted_count = 0
        for url in urls:
            if self.delete_course_by_url(url):
                deleted_count += 1
        return deleted_count
    
    def add_frontend_courses(self) -> bool:
        """
        Add curated frontend/web development courses to knowledge base.
        Includes 30 courses covering HTML, CSS, JavaScript, React, Angular, Vue, etc.
        
        Returns:
            True if successful, False otherwise
        """
        frontend_courses = [
            # FOUNDATIONAL HTML / CSS / JS COURSES (BEGINNER)
            {
                "title": "HTML & CSS Full Course",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/2022/responsive-web-design/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML", "CSS", "Responsive Design"],
                "description": "Foundational course covering HTML and CSS for responsive web design"
            },
            {
                "title": "JavaScript Algorithms & Data Structures",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JavaScript", "Algorithms", "Data Structures"],
                "description": "Foundational JavaScript course covering algorithms and data structures"
            },
            {
                "title": "Web Development for Beginners",
                "provider": "Microsoft",
                "url": "https://microsoft.github.io/Web-Dev-For-Beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML", "CSS", "JavaScript", "Web Development"],
                "description": "Foundational course covering HTML, CSS, and JavaScript for web development"
            },
            {
                "title": "Intro to HTML and CSS",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-html-and-css--ud001",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML", "CSS"],
                "description": "Foundational course introducing HTML and CSS"
            },
            {
                "title": "Intro to JavaScript",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-javascript--ud803",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JavaScript"],
                "description": "Foundational course introducing JavaScript programming"
            },
            {
                "title": "Web Development 101",
                "provider": "The Odin Project",
                "url": "https://www.theodinproject.com/paths/foundations/courses/foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML", "CSS", "JavaScript", "Web Development"],
                "description": "Foundational course covering web development basics"
            },
            {
                "title": "HTML, CSS, JavaScript for Web Developers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/html-css-javascript-for-web-developers",
                "difficulty": "Beginner",
                "duration": "4 weeks",
                "type": "course",
                "skills": ["HTML", "CSS", "JavaScript", "Web Development"],
                "description": "Foundational course covering HTML, CSS, and JavaScript for web developers"
            },
            {
                "title": "Programming Foundations with JavaScript, HTML & CSS",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/duke-programming-web",
                "difficulty": "Beginner",
                "duration": "4 weeks",
                "type": "course",
                "skills": ["JavaScript", "HTML", "CSS", "Programming"],
                "description": "Foundational course covering programming foundations with JavaScript, HTML, and CSS"
            },
            {
                "title": "Learn HTML",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/learn/learn-html",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML"],
                "description": "Foundational course for learning HTML"
            },
            {
                "title": "Learn CSS",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/learn/learn-css",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CSS"],
                "description": "Foundational course for learning CSS"
            },
            # RESPONSIVE DESIGN & UI/UX (INTERMEDIATE)
            {
                "title": "Responsive Web Design Bootcamp",
                "provider": "Scrimba",
                "url": "https://scrimba.com/learn/responsive",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Responsive Design", "CSS", "Web Design"],
                "description": "Intermediate course covering responsive web design and CSS"
            },
            {
                "title": "Modern HTML & CSS from the Beginning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/modern-html-css-from-the-beginning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML", "CSS", "Modern Web Development"],
                "description": "Intermediate course covering modern HTML and CSS development"
            },
            {
                "title": "Build Responsive Real-World Websites",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/design-and-develop-a-killer-website-with-html5-and-css3/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HTML5", "CSS3", "Responsive Design", "Web Development"],
                "description": "Intermediate course for building responsive real-world websites with HTML5 and CSS3"
            },
            {
                "title": "Google UX Design Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-ux-design",
                "difficulty": "Intermediate",
                "duration": "6 months",
                "type": "course",
                "skills": ["UX Design", "UI Design", "User Research", "Prototyping"],
                "description": "Intermediate professional certificate covering UX design, UI design, user research, and prototyping"
            },
            {
                "title": "Web Design for Everybody Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/web-design",
                "difficulty": "Intermediate",
                "duration": "4 months",
                "type": "course",
                "skills": ["Web Design", "HTML", "CSS", "JavaScript", "Responsive Design"],
                "description": "Intermediate specialization covering web design, HTML, CSS, JavaScript, and responsive design"
            },
            # JAVASCRIPT SPECIALIZATION
            {
                "title": "JavaScript: The Advanced Concepts",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/advanced-javascript-concepts/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JavaScript", "Advanced JavaScript", "ES6+", "Async Programming"],
                "description": "Advanced course covering advanced JavaScript concepts, ES6+, and async programming"
            },
            {
                "title": "The Modern JavaScript Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/javascript-beginners-complete-tutorial/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JavaScript", "Modern JavaScript", "ES6+"],
                "description": "Intermediate course covering modern JavaScript and ES6+ features"
            },
            {
                "title": "JavaScript30 (30 Projects in 30 Days)",
                "provider": "Wes Bos",
                "url": "https://javascript30.com/",
                "difficulty": "Intermediate",
                "duration": "30 days",
                "type": "course",
                "skills": ["JavaScript", "Projects", "Hands-on Learning"],
                "description": "Intermediate course with 30 JavaScript projects for hands-on learning"
            },
            {
                "title": "Eloquent JavaScript (Interactive Version)",
                "provider": "Eloquent JavaScript",
                "url": "https://eloquentjavascript.net/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "book",
                "skills": ["JavaScript", "Programming", "Computer Science"],
                "description": "Intermediate interactive book covering JavaScript programming and computer science concepts"
            },
            # FRONT-END FRAMEWORKS (REACT, ANGULAR, VUE)
            {
                "title": "React – The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["React.js", "Redux", "JavaScript", "Frontend Development"],
                "description": "Intermediate course covering React.js, Redux, and frontend development"
            },
            {
                "title": "Meta React Developer Certification",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/meta-react-native-developer",
                "difficulty": "Intermediate",
                "duration": "7 months",
                "type": "course",
                "skills": ["React.js", "React Native", "JavaScript", "Mobile Development"],
                "description": "Intermediate professional certificate covering React.js, React Native, and mobile development"
            },
            {
                "title": "Front-End Development Libraries Certification",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/front-end-development-libraries/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["React.js", "Bootstrap", "jQuery", "SASS", "Redux"],
                "description": "Intermediate certification covering front-end development libraries including React.js, Bootstrap, jQuery, SASS, and Redux"
            },
            {
                "title": "Angular – The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-guide-to-angular-2/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Angular", "TypeScript", "Frontend Development"],
                "description": "Intermediate course covering Angular framework, TypeScript, and frontend development"
            },
            {
                "title": "Vue.js 3 Complete Course",
                "provider": "Vue Mastery",
                "url": "https://www.vuemastery.com/courses/intro-to-vue-3/intro-to-vue3",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Vue.js", "JavaScript", "Frontend Development"],
                "description": "Intermediate course covering Vue.js 3, JavaScript, and frontend development"
            },
            {
                "title": "Svelte Tutorial",
                "provider": "Svelte.dev",
                "url": "https://svelte.dev/tutorial/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Svelte", "JavaScript", "Frontend Development"],
                "description": "Intermediate tutorial covering Svelte framework, JavaScript, and frontend development"
            },
            # ADVANCED FRONT-END / PROFESSIONAL PATHS
            {
                "title": "Meta Front-End Developer Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/meta-front-end-developer",
                "difficulty": "Advanced",
                "duration": "7 months",
                "type": "course",
                "skills": ["React.js", "JavaScript", "HTML", "CSS", "Frontend Development", "UI/UX"],
                "description": "Advanced professional certificate covering React.js, JavaScript, HTML, CSS, frontend development, and UI/UX"
            },
            {
                "title": "IBM Front-End Developer Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/ibm-frontend-developer",
                "difficulty": "Advanced",
                "duration": "4 months",
                "type": "course",
                "skills": ["Frontend Development", "React.js", "JavaScript", "Web Development"],
                "description": "Advanced professional certificate covering frontend development, React.js, JavaScript, and web development"
            },
            {
                "title": "Google Front-End Web Development",
                "provider": "Google",
                "url": "https://grow.google/certificates/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Frontend Development", "HTML", "CSS", "JavaScript"],
                "description": "Intermediate course covering frontend web development, HTML, CSS, and JavaScript"
            },
            {
                "title": "Full Modern Front-End Engineering Path",
                "provider": "The Odin Project",
                "url": "https://www.theodinproject.com/paths/full-stack-javascript",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JavaScript", "React.js", "Node.js", "Full Stack Development"],
                "description": "Advanced course path covering JavaScript, React.js, Node.js, and full stack development"
            },
            {
                "title": "Front-End Engineer Career Path",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/learn/paths/front-end-engineer-career-path",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Frontend Development", "React.js", "JavaScript", "HTML", "CSS"],
                "description": "Advanced career path covering frontend development, React.js, JavaScript, HTML, and CSS"
            }
        ]
        
        log.info(f"📚 Adding {len(frontend_courses)} frontend courses to knowledge base...")
        return self.batch_add_courses(frontend_courses)
    
    def add_ai_ml_courses(self) -> bool:
        """
        Add curated AI/ML courses to knowledge base.
        Includes 30 courses covering AI fundamentals, machine learning, deep learning, and MLOps.
        
        Returns:
            True if successful, False otherwise
        """
        ai_ml_courses = [
            {
                "title": "AI For Everyone",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ai-for-everyone",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI concepts", "Non-technical AI", "Ethical considerations"],
                "description": "Beginner course covering AI fundamentals, non-technical AI concepts, and ethical considerations"
            },
            {
                "title": "Machine Learning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/machine-learning",
                "difficulty": "Intermediate",
                "duration": "11 weeks",
                "type": "course",
                "skills": ["Supervised ML", "Regression", "SVMs", "Clustering", "Octave/Matlab"],
                "description": "Intermediate course covering classical machine learning including supervised ML, regression, SVMs, and clustering"
            },
            {
                "title": "Machine Learning with Python",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/machine-learning-with-python",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python ML", "scikit-learn", "Regression", "Classification"],
                "description": "Beginner course covering machine learning with Python using scikit-learn, regression, and classification"
            },
            {
                "title": "Python for Data Science, AI & Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-for-applied-data-science-ai",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "NumPy", "Pandas", "APIs"],
                "description": "Beginner course covering Python for data science, AI, and development including NumPy, Pandas, and APIs"
            },
            {
                "title": "Mathematics for Machine Learning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/mathematics-machine-learning",
                "difficulty": "Intermediate",
                "duration": "4 months",
                "type": "course",
                "skills": ["Linear algebra", "Multivariable calculus", "PCA"],
                "description": "Intermediate specialization covering mathematics for machine learning including linear algebra, multivariable calculus, and PCA"
            },
            {
                "title": "Fundamentals of Machine Learning",
                "provider": "AWS",
                "url": "https://www.aws.training",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ML workflow", "Cloud ML"],
                "description": "Beginner course covering ML foundations, ML workflow, and cloud ML"
            },
            {
                "title": "What is Data Science?",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/what-is-data-science",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data lifecycle", "ML basics"],
                "description": "Beginner course covering data science overview, data lifecycle, and ML basics"
            },
            {
                "title": "Introduction to AI",
                "provider": "edX",
                "url": "https://www.edx.org/learn/artificial-intelligence",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Search", "Reasoning", "AI basics"],
                "description": "Beginner course covering artificial intelligence including search, reasoning, and AI basics"
            },
            {
                "title": "Machine Learning Pipelines with Azure ML",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/azure-machine-learning-pipelines",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Azure ML Studio", "Pipeline automation", "MLOps"],
                "description": "Intermediate course covering MLOps, ML pipelines, Azure ML Studio, and pipeline automation"
            },
            {
                "title": "Nuts & Bolts of Machine Learning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/nuts-and-bolts-of-machine-learning",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ML lifecycle", "Data splitting", "Overfitting"],
                "description": "Beginner course covering ML foundations, ML lifecycle, data splitting, and overfitting"
            },
            {
                "title": "Machine Learning with Scikit-learn, PyTorch & Hugging Face",
                "provider": "Coursera",
                "url": "https://www.coursera.org/projects/machine-learning-huggingface",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PyTorch", "Transformers", "scikit-learn"],
                "description": "Intermediate course covering ML and DL frameworks including PyTorch, Transformers, and scikit-learn"
            },
            {
                "title": "MIT Data Science & ML Program",
                "provider": "MIT IDSS",
                "url": "https://idss-gl.mit.edu",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Deep Learning", "ML theory", "ML optimization"],
                "description": "Advanced professional certificate covering ML, deep learning, data science, ML theory, and ML optimization"
            },
            {
                "title": "AI & ML Courses",
                "provider": "Great Learning",
                "url": "https://www.mygreatlearning.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ML basics", "Python", "Deep Learning"],
                "description": "Intermediate courses covering AI, ML, and deep learning including ML basics, Python, and DL"
            },
            {
                "title": "Intro to Artificial Intelligence",
                "provider": "edX",
                "url": "https://www.edx.org/learn/artificial-intelligence",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Search", "Logical reasoning"],
                "description": "Beginner course covering AI including search and logical reasoning"
            },
            {
                "title": "AI Courses",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/category/artificial-intelligence",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python ML", "GenAI basics"],
                "description": "Intermediate courses covering AI, ML, and GenAI including Python ML and GenAI basics"
            },
            {
                "title": "Machine Learning with Python",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/machine-learning-with-python/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["sklearn", "Regression", "Neural networks basics"],
                "description": "Beginner course covering machine learning with Python including sklearn, regression, and neural network basics"
            },
            {
                "title": "Deep Learning with Python",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["TensorFlow/Keras", "Neural networks"],
                "description": "Intermediate course covering deep learning with Python including TensorFlow/Keras and neural networks"
            },
            {
                "title": "Elements of AI",
                "provider": "University of Helsinki",
                "url": "https://www.elementsofai.com",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI reasoning", "Ethics"],
                "description": "Beginner course covering AI concepts including AI reasoning and ethics"
            },
            {
                "title": "Intro to Machine Learning",
                "provider": "Google Cloud",
                "url": "https://cloud.google.com/training",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GCP ML tools", "AutoML"],
                "description": "Intermediate course covering cloud ML including GCP ML tools and AutoML"
            },
            {
                "title": "AI Courses",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/catalog/subject/artificial-intelligence",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python AI", "ML basics"],
                "description": "Beginner courses covering ML, AI, and Python including Python AI and ML basics"
            },
            {
                "title": "Applied Machine Learning with Python",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-machine-learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["sklearn", "Regression", "SVM", "Pipelines"],
                "description": "Intermediate course covering applied ML with Python including sklearn, regression, SVM, and pipelines"
            },
            {
                "title": "IBM AI Engineering Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/ai-engineer",
                "difficulty": "Advanced",
                "duration": "6 months",
                "type": "course",
                "skills": ["Deep Learning", "TensorFlow", "ML pipelines", "Model deployment"],
                "description": "Advanced professional certificate covering ML, deep learning, model deployment, TensorFlow, and ML pipelines"
            },
            {
                "title": "Google AI Essentials",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/google-ai-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI workflow", "GenAI"],
                "description": "Beginner course covering modern AI fundamentals including AI workflow and GenAI"
            },
            {
                "title": "Deep Learning Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/deep-learning",
                "difficulty": "Advanced",
                "duration": "5 months",
                "type": "course",
                "skills": ["Neural networks", "CNNs", "RNNs", "Optimization"],
                "description": "Advanced specialization covering deep learning including neural networks, CNNs, RNNs, and optimization"
            },
            {
                "title": "Machine Learning by Kaggle",
                "provider": "Kaggle Learn",
                "url": "https://www.kaggle.com/learn/intro-to-machine-learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ML with Python", "XGBoost"],
                "description": "Intermediate course covering ML with Python and XGBoost"
            },
            {
                "title": "Intro to TensorFlow for Deep Learning",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["TensorFlow", "CNNs", "RNNs"],
                "description": "Intermediate course covering deep learning with TensorFlow including CNNs and RNNs"
            },
            {
                "title": "Intro to Machine Learning",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-machine-learning--ud120",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["sklearn", "Algorithms", "Pipelines"],
                "description": "Intermediate course covering ML including sklearn, algorithms, and pipelines"
            },
            {
                "title": "Advanced Machine Learning Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/aml",
                "difficulty": "Advanced",
                "duration": "7 months",
                "type": "course",
                "skills": ["Bayesian ML", "Optimization", "Reinforcement learning"],
                "description": "Advanced specialization covering ML theory and practice including Bayesian ML, optimization, and reinforcement learning"
            },
            {
                "title": "Fast.ai Practical Deep Learning",
                "provider": "Fast.ai",
                "url": "https://course.fast.ai",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PyTorch", "DL best practices"],
                "description": "Intermediate course covering practical deep learning with PyTorch and DL best practices"
            },
            {
                "title": "MLOps Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/mlops",
                "difficulty": "Advanced",
                "duration": "4 months",
                "type": "course",
                "skills": ["Model deployment", "ML pipelines", "CI/CD for ML"],
                "description": "Advanced specialization covering MLOps, deployment, ML pipelines, and CI/CD for ML"
            }
        ]
        
        log.info(f"📚 Adding {len(ai_ml_courses)} AI/ML courses to knowledge base...")
        return self.batch_add_courses(ai_ml_courses)
    
    def add_data_science_courses(self) -> bool:
        """
        Add curated Data Science courses to knowledge base.
        Includes 30 courses covering data science fundamentals, statistics, visualization, and big data.
        
        Returns:
            True if successful, False otherwise
        """
        data_science_courses = [
            {
                "title": "Data Science Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/jhu-data-science",
                "difficulty": "Intermediate",
                "duration": "8 months",
                "type": "course",
                "skills": ["R", "Statistics", "Modeling", "Data cleaning", "Regression"],
                "description": "Intermediate specialization covering full data science workflow including R, statistics, modeling, data cleaning, and regression"
            },
            {
                "title": "IBM Data Science Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/ibm-data-science",
                "difficulty": "Beginner",
                "duration": "3 months",
                "type": "course",
                "skills": ["Python", "SQL", "Pandas", "ML basics", "Data visualization"],
                "description": "Beginner professional certificate covering data science foundations with hands-on labs including Python, SQL, Pandas, ML basics, and data visualization"
            },
            {
                "title": "Google Data Analytics Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-data-analytics",
                "difficulty": "Beginner",
                "duration": "6 months",
                "type": "course",
                "skills": ["Excel", "SQL", "Dashboards", "Data cleaning"],
                "description": "Beginner professional certificate covering data analytics and data handling including Excel, SQL, dashboards, and data cleaning"
            },
            {
                "title": "Data Science for Everyone",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/courses/data-science-for-everyone",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DS lifecycle", "Data types", "Workflow"],
                "description": "Beginner course covering general data science concepts including DS lifecycle, data types, and workflow"
            },
            {
                "title": "Applied Data Science with Python Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/data-science-python",
                "difficulty": "Intermediate",
                "duration": "5 months",
                "type": "course",
                "skills": ["Pandas", "Matplotlib", "ML", "Data wrangling"],
                "description": "Intermediate specialization covering Python and applied data science including Pandas, Matplotlib, ML, and data wrangling"
            },
            {
                "title": "Data Scientist Career Path",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/learn/paths/data-scientist",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "SQL", "ML", "Statistics"],
                "description": "Intermediate career path covering Python-centric data science including Python, SQL, ML, and statistics"
            },
            {
                "title": "Data Science Courses",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "Pandas", "TensorFlow basics"],
                "description": "Beginner courses covering Python data science and ML including Python, Pandas, and TensorFlow basics"
            },
            {
                "title": "Python Data Science Handbook (Course Version)",
                "provider": "O'Reilly",
                "url": "https://learning.oreilly.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["NumPy", "Pandas", "Matplotlib", "ML"],
                "description": "Intermediate course covering applied data science in Python including NumPy, Pandas, Matplotlib, and ML"
            },
            {
                "title": "Data Science Nanodegree",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/data-scientist-nanodegree--nd025",
                "difficulty": "Advanced",
                "duration": "4 months",
                "type": "course",
                "skills": ["ML", "Deep Learning", "Statistics", "Deployment"],
                "description": "Advanced nanodegree covering industry-ready data science including ML, deep learning, statistics, and deployment"
            },
            {
                "title": "Statistics With Python Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/statistics-with-python",
                "difficulty": "Intermediate",
                "duration": "4 months",
                "type": "course",
                "skills": ["Probability", "Distributions", "Hypothesis testing"],
                "description": "Intermediate specialization covering statistics including probability, distributions, and hypothesis testing"
            },
            {
                "title": "SQL for Data Science",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/sql-for-data-science",
                "difficulty": "Beginner",
                "duration": "4 weeks",
                "type": "course",
                "skills": ["SQL basics", "Joins", "Aggregations"],
                "description": "Beginner course covering SQL for data science including SQL basics, joins, and aggregations"
            },
            {
                "title": "Data Visualization with Python",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-for-data-visualization",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Matplotlib", "Seaborn", "Plotly"],
                "description": "Intermediate course covering data visualization including Matplotlib, Seaborn, and Plotly"
            },
            {
                "title": "Data Analysis with Python",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pandas", "NumPy", "Cleaning", "Statistics"],
                "description": "Beginner course covering data analysis including Pandas, NumPy, data cleaning, and statistics"
            },
            {
                "title": "Big Data Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/big-data",
                "difficulty": "Intermediate",
                "duration": "7 months",
                "type": "course",
                "skills": ["Hadoop", "Spark", "NoSQL"],
                "description": "Intermediate specialization covering big data and data science including Hadoop, Spark, and NoSQL"
            },
            {
                "title": "Data Science Foundations",
                "provider": "edX",
                "url": "https://www.edx.org/professional-certificate/ibm-data-science",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "Statistics", "DS workflow"],
                "description": "Beginner professional certificate covering data science basics including Python, statistics, and DS workflow"
            },
            {
                "title": "Tableau Data Visualization Course",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/data-visualization-tableau",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tableau", "Storytelling", "Dashboards"],
                "description": "Beginner course covering dashboarding with Tableau including storytelling and dashboards"
            },
            {
                "title": "Data Mining Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/data-mining",
                "difficulty": "Intermediate",
                "duration": "6 months",
                "type": "course",
                "skills": ["Clustering", "Associations", "Predictive models"],
                "description": "Intermediate specialization covering data mining and ML including clustering, associations, and predictive models"
            },
            {
                "title": "Excel to MySQL: Analytics Techniques",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/excel-mysql",
                "difficulty": "Intermediate",
                "duration": "5 months",
                "type": "course",
                "skills": ["SQL", "Excel", "Modeling"],
                "description": "Intermediate specialization covering analytics and data science including SQL, Excel, and modeling"
            },
            {
                "title": "Machine Learning for Data Science",
                "provider": "edX",
                "url": "https://www.edx.org/course/principles-machine-learning",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ML basics", "Regression", "Clustering"],
                "description": "Beginner course covering ML foundations for data science including ML basics, regression, and clustering"
            },
            {
                "title": "R Programming for Data Science",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/tracks/data-scientist-with-r",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["R", "Tidyverse", "Statistics"],
                "description": "Beginner track covering R for data science including R, tidyverse, and statistics"
            },
            {
                "title": "Python for Data Science",
                "provider": "UC Berkeley",
                "url": "https://extension.berkeley.edu",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "Data prep", "Exploration"],
                "description": "Beginner course covering Python and data science including Python, data preparation, and exploration"
            },
            {
                "title": "Data Science Bootcamp",
                "provider": "Springboard",
                "url": "https://www.springboard.com/courses/data-science-career-track/",
                "difficulty": "Advanced",
                "duration": "6 months",
                "type": "course",
                "skills": ["Python", "ML", "SQL", "Project portfolio"],
                "description": "Advanced bootcamp covering full data science training including Python, ML, SQL, and project portfolio"
            },
            {
                "title": "Applied Plotting, Charting & Data Representation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-plotting",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Matplotlib", "Design principles"],
                "description": "Beginner course covering data visualization including Matplotlib and design principles"
            },
            {
                "title": "Intro to Data Science",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-data-science--ud359",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "SQL", "Visualization"],
                "description": "Beginner course covering data science basics including Python, SQL, and visualization"
            },
            {
                "title": "Statistics for Data Science",
                "provider": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/news/tag/statistics/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "Statistics for DS"],
                "description": "Beginner course covering statistics including probability and statistics for data science"
            },
            {
                "title": "Data Exploration & Visualization with Python",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/courses",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pandas", "Visualization", "EDA"],
                "description": "Beginner course covering data visualization including Pandas, visualization, and exploratory data analysis"
            },
            {
                "title": "AWS Data Analytics Specialty Courses",
                "provider": "AWS",
                "url": "https://aws.amazon.com/training/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Redshift", "Glue", "Big data"],
                "description": "Advanced courses covering cloud analytics including Redshift, Glue, and big data"
            },
            {
                "title": "MITx Statistics and Data Science MicroMasters",
                "provider": "edX",
                "url": "https://micromasters.mit.edu/ds/",
                "difficulty": "Advanced",
                "duration": "12 months",
                "type": "course",
                "skills": ["Probability", "Inference", "ML theory"],
                "description": "Advanced MicroMasters covering statistics and ML including probability, inference, and ML theory"
            },
            {
                "title": "Harvard Data Science Professional Certificate",
                "provider": "edX",
                "url": "https://www.edx.org/professional-certificate/harvardx-data-science",
                "difficulty": "Intermediate",
                "duration": "8 months",
                "type": "course",
                "skills": ["R", "ML basics", "Statistics", "Data wrangling"],
                "description": "Intermediate professional certificate covering data science foundations including R, ML basics, statistics, and data wrangling"
            }
        ]
        
        log.info(f"📚 Adding {len(data_science_courses)} Data Science courses to knowledge base...")
        return self.batch_add_courses(data_science_courses)
    
    def add_cybersecurity_courses(self) -> bool:
        """
        Add curated Cybersecurity courses to knowledge base.
        Includes 30 courses covering cybersecurity fundamentals, ethical hacking, network security, and cloud security.
        
        Returns:
            True if successful, False otherwise
        """
        cybersecurity_courses = [
            {
                "title": "IBM Cybersecurity Analyst Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SIEM", "SOC", "Threat Analysis", "Incident Response", "Network Security"],
                "description": "Professional certificate covering cybersecurity analyst skills including SIEM, SOC operations, threat analysis, and incident response"
            },
            {
                "title": "Google Cybersecurity Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-cybersecurity",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linux", "SQL", "SIEM Tools", "Threat Detection"],
                "description": "Entry-level professional certificate covering cybersecurity fundamentals including Linux, SQL, SIEM tools, and threat detection"
            },
            {
                "title": "Introduction to Cybersecurity",
                "provider": "Cisco Networking Academy",
                "url": "https://www.netacad.com/courses/cybersecurity",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cyber Threats", "Attack Types", "Security Basics"],
                "description": "Free course covering cybersecurity fundamentals including cyber threats, attack types, and security basics"
            },
            {
                "title": "Cybersecurity Essentials",
                "provider": "Cisco Networking Academy",
                "url": "https://www.netacad.com/courses/cybersecurity",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Network Defense", "Firewalls", "Encryption"],
                "description": "Free course covering network security and defense including firewalls and encryption"
            },
            {
                "title": "CompTIA Security+ (SY0-701) Prep",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/securityplus/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Security Architecture", "Threats", "Risk Management"],
                "description": "Industry certification prep course covering security architecture, threats, and risk management for CompTIA Security+ exam"
            },
            {
                "title": "The Complete Cyber Security Course (Volume 1-4)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-cyber-security-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Anonymity", "Malware Defense", "Encryption"],
                "description": "Comprehensive practical cybersecurity course covering anonymity, malware defense, and encryption across 4 volumes"
            },
            {
                "title": "Cybersecurity for Everyone",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cyber-security",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Threats", "Vulnerabilities", "Basic Defense Measures"],
                "description": "Beginner course covering cybersecurity basics including threats, vulnerabilities, and basic defense measures"
            },
            {
                "title": "Introduction to Cyber Security",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/introduction-to-cyber-security",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Passwords", "Malware", "Network Basics"],
                "description": "Beginner course covering cybersecurity fundamentals including passwords, malware, and network basics"
            },
            {
                "title": "Cyber Security Basics",
                "provider": "edX",
                "url": "https://www.edx.org/course/cybersecurity-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CIA Triad", "Security Controls", "Threats"],
                "description": "Introductory course covering cyber concepts including CIA triad, security controls, and threats"
            },
            {
                "title": "Cybersecurity Fundamentals",
                "provider": "edX",
                "url": "https://www.edx.org/learn/cybersecurity/ibm",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cyber Attacks", "Vulnerability Scanning"],
                "description": "Beginner course covering general cybersecurity including cyber attacks and vulnerability scanning"
            },
            {
                "title": "Fundamentals of Network Security",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/network-security",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Firewalls", "IDS", "VPNs", "Encryption"],
                "description": "Beginner course covering network security including firewalls, IDS, VPNs, and encryption"
            },
            {
                "title": "Cybersecurity Risk Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cybersecurity-risk-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Risk Assessment", "Mitigation Strategies"],
                "description": "Intermediate course covering risk frameworks including risk assessment and mitigation strategies"
            },
            {
                "title": "Ethical Hacking Essentials",
                "provider": "EC-Council",
                "url": "https://codered.eccouncil.org/course/ethical-hacking-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Reconnaissance", "Scanning", "Exploitation Basics"],
                "description": "Free course covering ethical hacking foundations including reconnaissance, scanning, and exploitation basics"
            },
            {
                "title": "Network Defense Essentials",
                "provider": "EC-Council",
                "url": "https://codered.eccouncil.org/course/network-defense-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IDS/IPS", "Firewalls", "Packet Analysis"],
                "description": "Free course covering defensive operations including IDS/IPS, firewalls, and packet analysis"
            },
            {
                "title": "Digital Forensics Essentials",
                "provider": "EC-Council",
                "url": "https://codered.eccouncil.org/course/digital-forensics-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Forensic Imaging", "Evidence Handling"],
                "description": "Free course covering digital forensics including forensic imaging and evidence handling"
            },
            {
                "title": "TryHackMe — Complete Beginner Path",
                "provider": "TryHackMe",
                "url": "https://tryhackme.com/path/outline/beginner",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linux", "Networking", "Hacking Labs"],
                "description": "Practical cybersecurity labs covering Linux, networking, and hands-on hacking exercises with completion badge"
            },
            {
                "title": "TryHackMe — Offensive Pentesting Path",
                "provider": "TryHackMe",
                "url": "https://tryhackme.com/path/outline/pentesting",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Web Exploits", "Privilege Escalation", "Enumeration"],
                "description": "Intermediate course covering penetration testing including web exploits, privilege escalation, and enumeration with completion badge"
            },
            {
                "title": "Cybrary SOC Analyst Level 1",
                "provider": "Cybrary",
                "url": "https://www.cybrary.it/course/soc-analyst-level-1/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SIEM Tools", "Alerts", "Triage"],
                "description": "Course covering SOC operations including SIEM tools, alerts, and triage for SOC analyst roles"
            },
            {
                "title": "Cybrary Penetration Testing & Ethical Hacking",
                "provider": "Cybrary",
                "url": "https://www.cybrary.it/course/ethical-hacking/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Kali Linux", "Exploitation", "Reconnaissance"],
                "description": "Intermediate course covering penetration testing including Kali Linux, exploitation, and reconnaissance"
            },
            {
                "title": "Offensive Security Certified Professional (OSCP) Prep",
                "provider": "OffSec",
                "url": "https://www.offsec.com/courses/pen-200/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Exploitation", "Buffer Overflows", "Pivoting"],
                "description": "Advanced course covering penetration testing including exploitation, buffer overflows, and pivoting for OSCP certification"
            },
            {
                "title": "Certified Ethical Hacker (CEH) Course",
                "provider": "EC-Council",
                "url": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Web Hacking", "Network Attacks", "Cryptography"],
                "description": "Advanced course covering ethical hacking including web hacking, network attacks, and cryptography for CEH certification"
            },
            {
                "title": "Introduction to Information Security",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/intro-to-information-security--ud459",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CIA Triad", "Secure Design", "Threat Modeling"],
                "description": "Beginner course covering security principles including CIA triad, secure design, and threat modeling"
            },
            {
                "title": "Applied Cryptography",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cryptography",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Encryption", "Hashing", "Secure Protocols"],
                "description": "Intermediate course covering cryptography including encryption, hashing, and secure protocols"
            },
            {
                "title": "Introduction to Cybersecurity Tools & Cyber Attacks",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cybersecurity-tools",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SIEM", "IDS", "Firewalls"],
                "description": "Beginner course covering cyber tools overview including SIEM, IDS, and firewalls"
            },
            {
                "title": "Linux Security Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/linux-security/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linux Hardening", "Permissions", "SSH Security"],
                "description": "Course covering Linux for security including Linux hardening, permissions, and SSH security"
            },
            {
                "title": "Introduction to Network Security",
                "provider": "edX",
                "url": "https://www.edx.org/learn/network-security",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Firewalls", "VPNs", "Secure Routing"],
                "description": "Beginner course covering network defense including firewalls, VPNs, and secure routing"
            },
            {
                "title": "AWS Security Fundamentals",
                "provider": "AWS Training",
                "url": "https://aws.amazon.com/training/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IAM", "Cloud Threat Models"],
                "description": "Intermediate course covering cloud security including IAM and cloud threat models with free badge"
            },
            {
                "title": "Google Cloud Security Engineer Training",
                "provider": "Google Cloud",
                "url": "https://cloud.google.com/training",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Identity", "KMS", "Secure Architecture"],
                "description": "Advanced course covering cloud security and IAM including identity, KMS, and secure architecture"
            },
            {
                "title": "Introduction to Dark Web, Anonymity & Cryptocurrency",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/dark-web-anonymity/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["TOR", "OPSEC", "Anonymity", "Crypto Security"],
                "description": "Beginner course covering privacy and anonymity including TOR, OPSEC, anonymity, and crypto security"
            },
            {
                "title": "OWASP Top 10 Web Security Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/owasp-top-10/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["XSS", "SQLi", "CSRF", "Broken Auth", "Secure Coding"],
                "description": "Course covering web application security including XSS, SQLi, CSRF, broken auth, and secure coding based on OWASP Top 10"
            }
        ]
        
        log.info(f"📚 Adding {len(cybersecurity_courses)} Cybersecurity courses to knowledge base...")
        return self.batch_add_courses(cybersecurity_courses)
    
    def add_blockchain_courses(self) -> bool:
        """
        Add curated Blockchain courses to knowledge base.
        Includes 30 courses covering blockchain fundamentals, smart contracts, Ethereum, DeFi, and Web3 development.
        
        Returns:
            True if successful, False otherwise
        """
        blockchain_courses = [
            {
                "title": "Blockchain Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/blockchain",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Distributed Systems", "Smart Contracts", "Decentralization"],
                "description": "Specialization covering core blockchain concepts including distributed systems, smart contracts, and decentralization"
            },
            {
                "title": "Blockchain Basics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/blockchain-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hashing", "Blocks", "Decentralization"],
                "description": "Foundational course covering blockchain concepts including hashing, blocks, and decentralization"
            },
            {
                "title": "Blockchain Revolution Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/blockchain-revolution",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Industry Use Cases", "Web3", "Decentralized Marketplaces"],
                "description": "Specialization covering business applications of blockchain including industry use cases, Web3, and decentralized marketplaces"
            },
            {
                "title": "Bitcoin and Cryptocurrency Technologies",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cryptocurrency",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Consensus", "Blockchain Security", "Wallets"],
                "description": "Course covering Bitcoin architecture including consensus mechanisms, blockchain security, and wallets"
            },
            {
                "title": "Ethereum and Solidity: The Complete Developer's Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ethereum-and-solidity-the-complete-developers-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity", "DApps", "Web3.js"],
                "description": "Complete developer guide covering Ethereum smart contract development including Solidity, DApps, and Web3.js"
            },
            {
                "title": "Blockchain A–Z: Learn How to Build a Blockchain",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/build-your-blockchain-az/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python Blockchain", "Mining", "Nodes"],
                "description": "Course covering building blockchain from scratch including Python blockchain implementation, mining, and nodes"
            },
            {
                "title": "Certified Blockchain Developer (CBD)",
                "provider": "Blockchain Council",
                "url": "https://www.blockchain-council.org/certifications/certified-blockchain-developer/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Ethereum", "Hyperledger", "Use-case Development"],
                "description": "Professional certification covering blockchain development including Ethereum, Hyperledger, and use-case development"
            },
            {
                "title": "Certified Blockchain Expert (CBE)",
                "provider": "Blockchain Council",
                "url": "https://www.blockchain-council.org/certifications/certified-blockchain-expert/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Consensus", "Nodes", "Smart Contracts"],
                "description": "Certification covering blockchain architecture and use cases including consensus mechanisms, nodes, and smart contracts"
            },
            {
                "title": "Hyperledger Fabric for Developers",
                "provider": "edX",
                "url": "https://www.edx.org/learn/hyperledger",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Chaincode", "Fabric SDK", "Enterprise Blockchain"],
                "description": "Course covering Hyperledger Fabric including chaincode development, Fabric SDK, and enterprise blockchain"
            },
            {
                "title": "Blockchain for Business",
                "provider": "edX",
                "url": "https://www.edx.org/course/blockchain-for-business",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Enterprise Blockchain", "Permissioned Networks"],
                "description": "Course covering business blockchain use cases including enterprise blockchain and permissioned networks"
            },
            {
                "title": "Blockchain Developer Bootcamp",
                "provider": "ConsenSys Academy",
                "url": "https://consensys.net/academy/bootcamp/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity", "DApps", "Truffle", "Smart Contracts"],
                "description": "Bootcamp covering Ethereum development including Solidity, DApps, Truffle, and smart contracts"
            },
            {
                "title": "Introduction to Blockchain Technologies",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-blockchain-technologies",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Distributed Ledgers", "Smart Contracts"],
                "description": "Course covering blockchain fundamentals including distributed ledgers and smart contracts"
            },
            {
                "title": "Blockchain Developer Nanodegree",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/blockchain-developer-nanodegree--nd1309",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity", "DApps", "C++ Blockchain"],
                "description": "Nanodegree covering blockchain and Ethereum development including Solidity, DApps, and C++ blockchain"
            },
            {
                "title": "Solidity, Blockchain & Smart Contract Course",
                "provider": "freeCodeCamp",
                "url": "https://www.youtube.com/watch?v=M576WGiDBdQ",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity", "Remix", "Ethereum"],
                "description": "Free course covering Solidity programming including Remix IDE and Ethereum development"
            },
            {
                "title": "Blockchain Developer Program",
                "provider": "Simplilearn",
                "url": "https://www.simplilearn.com/blockchain-developer-certification-training",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Ethereum", "Hyperledger", "Smart Contracts"],
                "description": "Professional certification program covering blockchain development including Ethereum, Hyperledger, and smart contracts"
            },
            {
                "title": "Ethereum Blockchain Developer Bootcamp With Solidity",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ethereum-bootcamp/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Truffle", "Ganache", "Solidity"],
                "description": "Bootcamp covering Ethereum development including Truffle framework, Ganache, and Solidity"
            },
            {
                "title": "Blockchain Fundamentals",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/blockchain-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hashing", "Chaining", "Cryptography"],
                "description": "Course covering basics of blockchain including hashing, chaining, and cryptography"
            },
            {
                "title": "Smart Contracts in Solidity",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/smart-contracts/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity Syntax", "Events", "Modifiers"],
                "description": "Course covering Solidity smart contracts including syntax, events, and modifiers"
            },
            {
                "title": "Blockchain: Foundations and Use Cases",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/blockchain-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Decentralization", "Tokens", "Governance"],
                "description": "Course covering Web3 foundations including decentralization, tokens, and governance"
            },
            {
                "title": "Introduction to Web3",
                "provider": "Alchemy Academy",
                "url": "https://university.alchemy.com/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Wallets", "RPC", "Smart Contracts"],
                "description": "Free course covering Web3 developer track including wallets, RPC, and smart contracts"
            },
            {
                "title": "Smart Contract Security",
                "provider": "ConsenSys",
                "url": "https://consensys.net/academy/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Reentrancy", "Vulnerabilities", "Audits"],
                "description": "Course covering smart contract auditing including reentrancy attacks, vulnerabilities, and security audits"
            },
            {
                "title": "Zero to Mastery Blockchain Developer Course",
                "provider": "ZTM Academy",
                "url": "https://zerotomastery.io/courses/blockchain-developer/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solidity", "Web3.js", "dApp Development"],
                "description": "Course covering blockchain development including Solidity, Web3.js, and dApp development"
            },
            {
                "title": "Blockchain Certification Training",
                "provider": "Edureka",
                "url": "https://www.edureka.co/blockchain-training",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hyperledger", "Ethereum", "Cryptography"],
                "description": "Certification training covering blockchain fundamentals including Hyperledger, Ethereum, and cryptography"
            },
            {
                "title": "Build a Blockchain and Cryptocurrency",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/build-blockchain/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mining", "Proof-of-Work", "Flask API"],
                "description": "Course covering Python blockchain development including mining, Proof-of-Work consensus, and Flask API"
            },
            {
                "title": "DeFi (Decentralized Finance) MOOC",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/defi",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Liquidity Pools", "DEXs", "DeFi Protocols"],
                "description": "Course covering decentralized finance including liquidity pools, DEXs, and DeFi protocols"
            },
            {
                "title": "Understanding Solidity – Beginner to Advanced",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/solidity/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Smart Contracts", "Events", "Interfaces"],
                "description": "Course covering Solidity programming from beginner to advanced including smart contracts, events, and interfaces"
            },
            {
                "title": "Web3 Masterclass",
                "provider": "Moralis Academy",
                "url": "https://academy.moralis.io/courses",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Web3.js", "Moralis API", "NFTs"],
                "description": "Masterclass covering Web3 development including Web3.js, Moralis API, and NFTs"
            },
            {
                "title": "NFT Development for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/nft-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ERC-721", "Metadata", "NFT Minting"],
                "description": "Course covering blockchain NFT development including ERC-721 standard, metadata, and NFT minting"
            },
            {
                "title": "Blockchain Programming Using Python",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/topics/blockchain",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hashing", "PoW", "Python Blockchain"],
                "description": "Course covering blockchain implementation using Python including hashing, Proof-of-Work, and Python blockchain"
            },
            {
                "title": "Foundations of Decentralized Applications (DApps)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/courses?query=dapps",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Smart Contracts", "dApp Patterns", "Web3 Integration"],
                "description": "Course covering DApp architecture including smart contracts, dApp patterns, and Web3 integration"
            }
        ]
        
        log.info(f"📚 Adding {len(blockchain_courses)} Blockchain courses to knowledge base...")
        return self.batch_add_courses(blockchain_courses)
    
    def add_advanced_frontend_courses(self) -> bool:
        """
        Add advanced Frontend Development courses to knowledge base.
        Includes 10 courses covering modern frontend technologies, CSS frameworks, and UI development.
        
        Returns:
            True if successful, False otherwise
        """
        frontend_courses = [
            {
                "title": "Complete Guide to Tailwind CSS",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/tailwindcss-from-scratch/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tailwind", "Responsive UI", "CSS Optimization"],
                "description": "Course covering utility-first CSS framework including Tailwind, responsive UI design, and CSS optimization"
            },
            {
                "title": "Advanced CSS and Sass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/advanced-css-and-sass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Flexbox", "Grid", "Animations", "BEM"],
                "description": "Course covering modern CSS architecture including Flexbox, Grid, animations, and BEM methodology"
            },
            {
                "title": "Vue.js Essentials",
                "provider": "Codecademy",
                "url": "https://www.codecademy.com/learn/learn-vue-js",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Components", "State", "Routing"],
                "description": "Course covering Vue.js basics including components, state management, and routing"
            },
            {
                "title": "React + Vite Frontend Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/react-vite/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["React Hooks", "Vite", "APIs"],
                "description": "Course covering React with modern tooling including React hooks, Vite build tool, and API integration"
            },
            {
                "title": "SASS / SCSS Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sasscourse/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mixins", "Nesting", "Variables"],
                "description": "Course covering CSS preprocessors including SASS/SCSS mixins, nesting, and variables"
            },
            {
                "title": "Building Web Interfaces with Svelte",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/svelte",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["State", "Reactivity", "UI Components"],
                "description": "Course covering Svelte UI framework including state management, reactivity, and UI components"
            },
            {
                "title": "Web Accessibility Course",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/web-accessibility--ud891",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["WCAG", "ARIA Roles"],
                "description": "Course covering accessibility (A11y) including WCAG guidelines and ARIA roles"
            },
            {
                "title": "JAMstack Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/jamstack/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Next.js", "Netlify", "Static Generation"],
                "description": "Course covering modern static frontends including Next.js, Netlify, and static site generation"
            },
            {
                "title": "React Native for Web",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/react-native",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Universal Components", "Expo"],
                "description": "Course covering cross-platform UI development including universal components and Expo framework"
            },
            {
                "title": "CSS Animations & Transitions Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/css-animation-transitions/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Keyframes", "Transforms"],
                "description": "Course covering animation for UI including CSS keyframes and transforms"
            }
        ]
        
        log.info(f"📚 Adding {len(frontend_courses)} Advanced Frontend courses to knowledge base...")
        return self.batch_add_courses(frontend_courses)
    
    def add_backend_courses(self) -> bool:
        """
        Add Backend Development courses to knowledge base.
        Includes 10 courses covering various backend technologies and frameworks.
        
        Returns:
            True if successful, False otherwise
        """
        backend_courses = [
            {
                "title": "Node.js Microservices",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/nodejs-microservices/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Docker", "Messaging Queues"],
                "description": "Course covering distributed backend architecture including Docker and messaging queues"
            },
            {
                "title": "Django Advanced — Building APIs",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/django-advanced/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DRF", "Authentication"],
                "description": "Course covering Django REST development including Django REST Framework and authentication"
            },
            {
                "title": "Java Spring Boot Microservices",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/spring-boot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spring Security", "REST", "JPA"],
                "description": "Course covering Java backend development including Spring Security, REST APIs, and JPA"
            },
            {
                "title": "Go Backend Development",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/golang/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Goroutines", "net/http"],
                "description": "Course covering Go services including goroutines and net/http package"
            },
            {
                "title": "Rust Backend Essentials",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/rust-backend/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Actix", "Concurrency"],
                "description": "Course covering Rust backend development including Actix framework and concurrency"
            },
            {
                "title": "Laravel APIs and Backend Architecture",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/laravel-api/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Eloquent", "Routing", "JWT"],
                "description": "Course covering PHP backend development including Eloquent ORM, routing, and JWT authentication"
            },
            {
                "title": "ASP.NET Core Web API",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/aspdotnet-core",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["C#", "Entity Framework"],
                "description": "Course covering .NET backend development including C# and Entity Framework"
            },
            {
                "title": "Build REST APIs with FastAPI",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/fastapi/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Async API", "Validation", "JWT"],
                "description": "Course covering Python FastAPI including async API development, validation, and JWT authentication"
            },
            {
                "title": "PostgreSQL Backend Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/postgresql-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Relational Design", "Queries"],
                "description": "Course covering backend development with SQL including relational database design and queries"
            },
            {
                "title": "Node.js + GraphQL Server",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/nodejs-graphql/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Apollo Server", "Resolvers"],
                "description": "Course covering GraphQL backend including Apollo Server and resolvers"
            }
        ]
        
        log.info(f"📚 Adding {len(backend_courses)} Backend courses to knowledge base...")
        return self.batch_add_courses(backend_courses)
    
    def add_fullstack_courses(self) -> bool:
        """
        Add Full-Stack Development courses to knowledge base.
        Includes 10 courses covering various full-stack technology combinations.
        
        Returns:
            True if successful, False otherwise
        """
        fullstack_courses = [
            {
                "title": "MERN Stack Zero to Hero",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mern-stack-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["MongoDB", "Express", "React", "Node"],
                "description": "Course covering MERN stack development including MongoDB, Express, React, and Node.js"
            },
            {
                "title": "Full-Stack Web Development (Ruby on Rails + JS)",
                "provider": "The Odin Project",
                "url": "https://www.theodinproject.com/paths/full-stack-ruby-on-rails",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Rails", "JavaScript", "MVC"],
                "description": "Free course covering Rails and frontend development including Rails framework, JavaScript, and MVC architecture"
            },
            {
                "title": "Django + React Full-Stack",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/django-react/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DRF", "JWT", "React UI"],
                "description": "Course covering full-stack Python development including Django REST Framework, JWT, and React UI"
            },
            {
                "title": "Next.js + Node Full-Stack",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/nextjs-node/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SSR", "API Routes"],
                "description": "Course covering modern full-stack development including server-side rendering and API routes"
            },
            {
                "title": "Laravel + Vue Full-Stack Web Apps",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/laravel-vue/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Eloquent", "Routing", "SPA"],
                "description": "Course covering PHP and Vue full-stack development including Eloquent ORM, routing, and single-page applications"
            },
            {
                "title": ".NET + Angular Full-Stack",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/paths/full-stack-dotnet",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Entity Framework", "Angular"],
                "description": "Course covering C# and Angular full-stack development including Entity Framework and Angular framework"
            },
            {
                "title": "React Native + Node Full-Stack",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/react-native",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["React Native", "Node.js", "Mobile Backend"],
                "description": "Course covering mobile full-stack development including React Native and Node.js backend"
            },
            {
                "title": "Go + React Full-Stack",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/golang-react/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Golang", "React", "Backend"],
                "description": "Course covering Golang backend and React frontend full-stack development"
            },
            {
                "title": "Java Spring Boot + React",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/spring-boot-react/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spring Boot", "React", "Enterprise"],
                "description": "Course covering enterprise full-stack development including Spring Boot and React"
            },
            {
                "title": "Firebase Full-Stack Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/firebase",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Firebase", "Serverless", "Real-time"],
                "description": "Course covering serverless full-stack development including Firebase services and real-time features"
            }
        ]
        
        log.info(f"📚 Adding {len(fullstack_courses)} Full-Stack courses to knowledge base...")
        return self.batch_add_courses(fullstack_courses)
    
    def add_mobile_development_courses(self) -> bool:
        """
        Add Mobile App Development courses to knowledge base.
        Includes 10 courses covering iOS, Android, and cross-platform mobile development.
        
        Returns:
            True if successful, False otherwise
        """
        mobile_courses = [
            {
                "title": "iOS & Swift — Complete App Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ios-bootcamp/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Swift", "iOS", "UIKit"],
                "description": "Course covering Swift and iOS app development including UIKit framework"
            },
            {
                "title": "Android App Development with Kotlin",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/android-kotlin/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Kotlin", "Android", "Jetpack"],
                "description": "Course covering Android Kotlin app development including Jetpack libraries"
            },
            {
                "title": "Flutter App Development Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/flutter-bootcamp/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Flutter", "Dart", "Cross-platform"],
                "description": "Course covering Flutter app development including Dart programming and cross-platform development"
            },
            {
                "title": "React Native Zero to Mastery",
                "provider": "ZTM Academy",
                "url": "https://zerotomastery.io/courses/react-native/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["React Native", "JavaScript", "Mobile"],
                "description": "Course covering React Native mobile app development from beginner to advanced"
            },
            {
                "title": "SwiftUI Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/swiftui-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SwiftUI", "iOS", "Declarative UI"],
                "description": "Course covering SwiftUI framework for iOS app development including declarative UI patterns"
            },
            {
                "title": "Android Jetpack Compose Essentials",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/jetpack-compose/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Jetpack Compose", "Kotlin", "Modern UI"],
                "description": "Course covering Android Jetpack Compose for modern UI development"
            },
            {
                "title": "KMM (Kotlin Multiplatform Mobile)",
                "provider": "JetBrains Academy",
                "url": "https://hyperskill.org",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Kotlin", "Multiplatform", "Shared Code"],
                "description": "Course covering cross-platform Kotlin mobile development including shared codebase"
            },
            {
                "title": "Dart for Mobile Developers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/dart-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Dart", "Flutter", "Mobile"],
                "description": "Course covering Dart programming essentials for mobile developers"
            },
            {
                "title": "iOS Advanced Architecture (VIPER, MVVM)",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VIPER", "MVVM", "iOS Architecture"],
                "description": "Advanced course covering iOS architecture patterns including VIPER and MVVM"
            },
            {
                "title": "Android Apps with Firebase Backend",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/android-firebase/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Android", "Firebase", "Backend"],
                "description": "Course covering Android app development with Firebase backend integration"
            }
        ]
        
        log.info(f"📚 Adding {len(mobile_courses)} Mobile Development courses to knowledge base...")
        return self.batch_add_courses(mobile_courses)
    
    def add_devops_courses(self) -> bool:
        """
        Add DevOps / CI-CD courses to knowledge base.
        Includes 10 courses covering Docker, Kubernetes, CI/CD, and infrastructure automation.
        
        Returns:
            True if successful, False otherwise
        """
        devops_courses = [
            {
                "title": "Docker Mastery",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/docker-mastery/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Docker", "Containers", "Orchestration"],
                "description": "Course covering Docker fundamentals including containers and orchestration"
            },
            {
                "title": "Kubernetes for the Absolute Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/kubernetes-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Kubernetes", "K8s", "Container Orchestration"],
                "description": "Course covering Kubernetes basics including container orchestration"
            },
            {
                "title": "GitOps with ArgoCD",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/gitops/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GitOps", "ArgoCD", "CD"],
                "description": "Course covering GitOps practices including ArgoCD for continuous deployment"
            },
            {
                "title": "CI/CD Pipelines with GitHub Actions",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/github-actions/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GitHub Actions", "CI/CD", "Automation"],
                "description": "Course covering GitHub automation including CI/CD pipelines with GitHub Actions"
            },
            {
                "title": "Jenkins Pipeline as Code",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/jenkins-pipeline/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Jenkins", "Pipeline", "Automation"],
                "description": "Course covering Jenkins automation including pipeline as code"
            },
            {
                "title": "Terraform for DevOps Engineers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/terraform-beginner-to-advanced/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Terraform", "IaC", "Infrastructure"],
                "description": "Course covering Infrastructure as Code including Terraform for DevOps engineers"
            },
            {
                "title": "Ansible for Configuration Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ansible-advanced/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Ansible", "Automation", "Configuration"],
                "description": "Course covering automation and configuration management with Ansible"
            },
            {
                "title": "Monitoring with Prometheus & Grafana",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/prometheus-grafana/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Prometheus", "Grafana", "Monitoring"],
                "description": "Course covering monitoring and observability including Prometheus and Grafana"
            },
            {
                "title": "DevOps Zero to Hero",
                "provider": "Edureka",
                "url": "https://www.edureka.co/devops-certification-training",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DevOps", "CI/CD", "Automation"],
                "description": "Course covering DevOps fundamentals including CI/CD and automation practices"
            },
            {
                "title": "Linux for DevOps Engineers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/linux-devops/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linux", "Shell Scripting", "System Administration"],
                "description": "Course covering Linux fundamentals for DevOps engineers including shell scripting and system administration"
            }
        ]
        
        log.info(f"📚 Adding {len(devops_courses)} DevOps courses to knowledge base...")
        return self.batch_add_courses(devops_courses)
    
    def add_cloud_engineering_courses(self) -> bool:
        """
        Add Cloud Engineering courses to knowledge base.
        Includes 10 courses covering AWS, Azure, GCP, and cloud security.
        
        Returns:
            True if successful, False otherwise
        """
        cloud_courses = [
            {
                "title": "AWS Cloud Practitioner Essentials",
                "provider": "AWS Training",
                "url": "https://www.aws.training/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AWS", "Cloud Basics", "Services"],
                "description": "Course covering AWS basics including cloud fundamentals and core services"
            },
            {
                "title": "AWS Solutions Architect (SAA-C03)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AWS Architecture", "Solutions Design", "Best Practices"],
                "description": "Course covering AWS architecture including solutions design and best practices"
            },
            {
                "title": "AWS Lambda & Serverless Framework",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-lambda-serverless/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AWS Lambda", "Serverless", "Functions"],
                "description": "Course covering serverless computing including AWS Lambda and serverless framework"
            },
            {
                "title": "Azure Fundamentals (AZ-900)",
                "provider": "Microsoft Learn",
                "url": "https://learn.microsoft.com/training/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Azure", "Cloud Basics", "Services"],
                "description": "Course covering Azure basics including cloud fundamentals and core services"
            },
            {
                "title": "Azure DevOps Engineer (AZ-400)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/azure-devops/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Azure DevOps", "CI/CD", "Pipelines"],
                "description": "Course covering Azure DevOps including CI/CD pipelines and automation"
            },
            {
                "title": "Azure Administrator (AZ-104)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/az-104",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Azure Administration", "Operations", "Management"],
                "description": "Course covering Azure operations including administration and management"
            },
            {
                "title": "Google Cloud Digital Leader",
                "provider": "Google Cloud Skills Boost",
                "url": "https://www.cloudskillsboost.google",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GCP", "Cloud Fundamentals", "Services"],
                "description": "Course covering cloud fundamentals including Google Cloud Platform services"
            },
            {
                "title": "Google Cloud Associate Engineer",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gcp-associate-engineer",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GCP", "Engineering", "Infrastructure"],
                "description": "Course covering GCP engineer preparation including infrastructure and engineering practices"
            },
            {
                "title": "Terraform on AWS (Hands-On)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/terraform-aws/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Terraform", "AWS", "IaC"],
                "description": "Hands-on course covering Infrastructure as Code on AWS using Terraform"
            },
            {
                "title": "Cloud Security Fundamentals",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/cloud-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Security", "IAM", "Compliance"],
                "description": "Course covering cloud security fundamentals including IAM and compliance"
            }
        ]
        
        log.info(f"📚 Adding {len(cloud_courses)} Cloud Engineering courses to knowledge base...")
        return self.batch_add_courses(cloud_courses)
    
    def add_aws_courses(self) -> bool:
        """
        Add AWS-specific courses to knowledge base.
        Includes 10 courses covering AWS fundamentals, development, data engineering, and specialty certifications.
        
        Returns:
            True if successful, False otherwise
        """
        aws_courses = [
            {
                "title": "AWS Technical Essentials",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/aws-technical-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["EC2", "S3", "VPC", "IAM"],
                "description": "Course covering AWS fundamentals including EC2, S3, VPC, and IAM"
            },
            {
                "title": "AWS Developer Associate Prep",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-certification-developer-associate/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Lambda", "API Gateway", "DynamoDB"],
                "description": "Course covering AWS development including Lambda, API Gateway, and DynamoDB"
            },
            {
                "title": "AWS Data Engineering",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/aws-data-engineering",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Redshift", "Glue", "Kinesis"],
                "description": "Course covering AWS analytics stack including Redshift, Glue, and Kinesis"
            },
            {
                "title": "AWS Cloud DevOps Engineer",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/cloud-devops-nanodegree--nd9991",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CI/CD", "CloudFormation"],
                "description": "Course covering DevOps on AWS including CI/CD pipelines and CloudFormation"
            },
            {
                "title": "AWS Certified SysOps Administrator",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-sysops/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Monitoring", "Automation", "Scaling"],
                "description": "Course covering AWS operations including monitoring, automation, and scaling"
            },
            {
                "title": "AWS Advanced Networking",
                "provider": "A Cloud Guru",
                "url": "https://acloudguru.com",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VPC", "Direct Connect", "Transit Gateway"],
                "description": "Course covering networking on AWS including VPC, Direct Connect, and Transit Gateway"
            },
            {
                "title": "AWS Serverless API Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-serverless/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Lambda", "API Gateway", "SQS"],
                "description": "Course covering building serverless APIs including Lambda, API Gateway, and SQS"
            },
            {
                "title": "AWS Machine Learning Specialty",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/aws-machine-learning",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SageMaker", "Model Deployment"],
                "description": "Course covering ML on AWS including SageMaker and model deployment"
            },
            {
                "title": "AWS Database Specialty Training",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-database-specialty/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["RDS", "Aurora", "DynamoDB"],
                "description": "Course covering AWS databases including RDS, Aurora, and DynamoDB"
            },
            {
                "title": "AWS Cloud Practitioner Crash Course",
                "provider": "Simplilearn",
                "url": "https://www.simplilearn.com/aws-cloud-practitioner-certification-training",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Basic Cloud Concepts"],
                "description": "Course covering AWS basics including fundamental cloud concepts"
            }
        ]
        
        log.info(f"📚 Adding {len(aws_courses)} AWS courses to knowledge base...")
        return self.batch_add_courses(aws_courses)
    
    def add_azure_courses(self) -> bool:
        """
        Add Microsoft Azure courses to knowledge base.
        Includes 10 courses covering Azure fundamentals, development, architecture, and specialty certifications.
        
        Returns:
            True if successful, False otherwise
        """
        azure_courses = [
            {
                "title": "AZ-900: Azure Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/az900-azure/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Concepts", "Azure Services"],
                "description": "Course covering Azure basics including cloud concepts and Azure services"
            },
            {
                "title": "AZ-204: Azure Developer Associate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/microsoft-azure-dev",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Azure Functions", "API Management"],
                "description": "Course covering Azure development including Azure Functions and API Management"
            },
            {
                "title": "AZ-305: Azure Solutions Architect Expert",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/az-305-azure/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Governance", "Network Design"],
                "description": "Course covering architecture design including governance and network design"
            },
            {
                "title": "Azure DevOps Engineer Expert",
                "provider": "Microsoft Learn",
                "url": "https://learn.microsoft.com/training",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["YAML Pipelines", "Repos", "IaC"],
                "description": "Course covering DevOps including YAML pipelines, repos, and Infrastructure as Code"
            },
            {
                "title": "Azure Kubernetes Service (AKS) Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/azure-kubernetes-service/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AKS", "Scaling", "Deployments"],
                "description": "Course covering AKS including scaling and deployments"
            },
            {
                "title": "Azure Data Engineering (DP-203)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/microsoft-azure-data-engineering",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Synapse", "Data Factory"],
                "description": "Course covering data engineering including Synapse and Data Factory"
            },
            {
                "title": "Azure AI Engineer Associate",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cognitive Services", "Bot Framework"],
                "description": "Course covering AI on Azure including Cognitive Services and Bot Framework"
            },
            {
                "title": "Azure Networking Deep Dive",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/azure-networking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VNet", "VPN", "ExpressRoute"],
                "description": "Course covering Azure networking including VNet, VPN, and ExpressRoute"
            },
            {
                "title": "Azure Security Technologies (AZ-500)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/az-500-security/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IAM", "Defender", "Security Posture"],
                "description": "Course covering cloud security including IAM, Defender, and security posture"
            },
            {
                "title": "Azure SaaS App Development",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Multi-tenancy", "API Design"],
                "description": "Course covering SaaS architecture including multi-tenancy and API design"
            }
        ]
        
        log.info(f"📚 Adding {len(azure_courses)} Azure courses to knowledge base...")
        return self.batch_add_courses(azure_courses)
    
    def add_gcp_courses(self) -> bool:
        """
        Add Google Cloud Platform courses to knowledge base.
        Includes 10 courses covering GCP fundamentals, engineering, DevOps, and specialty certifications.
        
        Returns:
            True if successful, False otherwise
        """
        gcp_courses = [
            {
                "title": "Google Cloud Fundamentals: Core Infrastructure",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gcp-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Compute Engine", "IAM"],
                "description": "Course covering GCP essentials including Compute Engine and IAM"
            },
            {
                "title": "Associate Cloud Engineer Preparation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/cloud-engineering",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Deployment", "IAM", "Networking"],
                "description": "Course covering GCP certification including deployment, IAM, and networking"
            },
            {
                "title": "Google Cloud DevOps Engineer Path",
                "provider": "Google Cloud Skills Boost",
                "url": "https://www.cloudskillsboost.google",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SRE", "CI/CD", "Monitoring"],
                "description": "Course covering DevOps including SRE, CI/CD, and monitoring"
            },
            {
                "title": "Google Cloud Networking Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/google-cloud-networking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VPC", "Load Balancing"],
                "description": "Course covering GCP networks including VPC and load balancing"
            },
            {
                "title": "Building Scalable Apps with Firebase",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/firebase-cloud",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Firestore", "Authentication"],
                "description": "Course covering Firebase cloud apps including Firestore and authentication"
            },
            {
                "title": "GCP Data Engineering (Professional Certificate)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/gcp-data-engineering",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["BigQuery", "Dataflow"],
                "description": "Course covering data engineering including BigQuery and Dataflow"
            },
            {
                "title": "GCP Security Engineer Training",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/security-gcp",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["KMS", "IAM", "Secure Workloads"],
                "description": "Course covering cloud security including KMS, IAM, and secure workloads"
            },
            {
                "title": "Google Kubernetes Engine (GKE) Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gke-fundamentals",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Clusters", "Deployments"],
                "description": "Course covering GKE including clusters and deployments"
            },
            {
                "title": "Architecting with Google Kubernetes Engine",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/architecting-google-kubernetes-engine",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GKE", "Container Orchestration"],
                "description": "Course covering K8s on GCP including GKE and container orchestration"
            },
            {
                "title": "GCP Serverless Cloud Functions Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/google-cloud-functions/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Functions", "Pub/Sub"],
                "description": "Course covering serverless including Cloud Functions and Pub/Sub"
            }
        ]
        
        log.info(f"📚 Adding {len(gcp_courses)} GCP courses to knowledge base...")
        return self.batch_add_courses(gcp_courses)
    
    def add_cloud_architecture_courses(self) -> bool:
        """
        Add Cloud Architecture courses to knowledge base.
        Includes 10 courses covering cloud architecture patterns, design principles, and multi-cloud strategies.
        
        Returns:
            True if successful, False otherwise
        """
        architecture_courses = [
            {
                "title": "Cloud Architecture with Google",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/cloud-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Scalable Design", "HA"],
                "description": "Course covering architecture including scalable design and high availability"
            },
            {
                "title": "AWS Well-Architected Framework",
                "provider": "A Cloud Guru",
                "url": "https://acloudguru.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["5 Pillars", "Reliability"],
                "description": "Course covering AWS architecture including 5 Pillars and reliability"
            },
            {
                "title": "Azure Cloud Architect Expert Path",
                "provider": "Microsoft Learn",
                "url": "https://learn.microsoft.com/training",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Governance", "Networking"],
                "description": "Course covering Azure architecture including governance and networking"
            },
            {
                "title": "Cloud Systems Architecture (AWS)",
                "provider": "edX",
                "url": "https://www.edx.org",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Distributed Systems"],
                "description": "Course covering multi-tier cloud design including distributed systems"
            },
            {
                "title": "Cloud Native Architecture",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/cloud-native-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Microservices", "Containers"],
                "description": "Course covering cloud-native patterns including microservices and containers"
            },
            {
                "title": "Architecting Cloud Applications",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/cloud-applications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HA", "Caching", "Resilience"],
                "description": "Course covering software architecture including HA, caching, and resilience"
            },
            {
                "title": "Building Distributed Systems",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/distributed-systems/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Consistency", "Fault Tolerance"],
                "description": "Course covering distributed design including consistency and fault tolerance"
            },
            {
                "title": "Designing APIs for Cloud Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/api-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["REST", "gRPC"],
                "description": "Course covering API architecture including REST and gRPC"
            },
            {
                "title": "Multi-Cloud Architecture Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/multi-cloud-architecture/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HA", "Scalability"],
                "description": "Course covering AWS + Azure + GCP design including HA and scalability"
            },
            {
                "title": "Cloud Infrastructure & Services",
                "provider": "edX",
                "url": "https://www.edx.org/learn/cloud",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Virtualization", "Distributed Computing"],
                "description": "Course covering core cloud computing including virtualization and distributed computing"
            }
        ]
        
        log.info(f"📚 Adding {len(architecture_courses)} Cloud Architecture courses to knowledge base...")
        return self.batch_add_courses(architecture_courses)
    
    def add_cloud_security_courses(self) -> bool:
        """
        Add Cloud Security courses to knowledge base.
        Includes 10 courses covering cloud security fundamentals, specialty certifications, and advanced security practices.
        
        Returns:
            True if successful, False otherwise
        """
        security_courses = [
            {
                "title": "AWS Certified Security Specialty",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-security-specialty/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IAM", "KMS", "Audit"],
                "description": "Course covering AWS security including IAM, KMS, and audit"
            },
            {
                "title": "Azure Security Engineer (AZ-500)",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Defender", "SIEM"],
                "description": "Course covering Azure security including Defender and SIEM"
            },
            {
                "title": "Google Cloud Security Engineer",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/security-gcp",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["KMS", "IAM Policies"],
                "description": "Course covering GCP security including KMS and IAM policies"
            },
            {
                "title": "Cloud Security Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cloud-security/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Shared Responsibility", "IAM"],
                "description": "Course covering cloud security basics including shared responsibility model and IAM"
            },
            {
                "title": "CISSP Certification Training",
                "provider": "Simplilearn",
                "url": "https://www.simplilearn.com/cissp-certification-training",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Security Governance"],
                "description": "Course covering security leadership including security governance"
            },
            {
                "title": "Cloud Security Architecture",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses/cloud-security-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Threat Modeling", "Security Patterns"],
                "description": "Course covering secure cloud design including threat modeling and security patterns"
            },
            {
                "title": "Zero Trust Cloud Security",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/zero-trust/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IAM", "Segmentation"],
                "description": "Course covering Zero Trust including IAM and segmentation"
            },
            {
                "title": "SaaS Security Essentials",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/saas-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Protection", "Identity"],
                "description": "Course covering SaaS security including data protection and identity management"
            },
            {
                "title": "Kubernetes Security",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/kubernetes-security/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["RBAC", "Workload Isolation"],
                "description": "Course covering K8s security including RBAC and workload isolation"
            },
            {
                "title": "Cloud Penetration Testing",
                "provider": "Pentester Academy",
                "url": "https://www.pentesteracademy.com",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Exploitation", "IAM Bypass"],
                "description": "Course covering cloud offensive security including cloud exploitation and IAM bypass techniques"
            }
        ]
        
        log.info(f"📚 Adding {len(security_courses)} Cloud Security courses to knowledge base...")
        return self.batch_add_courses(security_courses)
    
    def add_python_courses(self) -> bool:
        """
        Add Python programming courses to knowledge base.
        Includes 10 courses covering Python fundamentals, OOP, automation, and specialized applications.
        
        Returns:
            True if successful, False otherwise
        """
        python_courses = [
            {
                "title": "Python for Everybody",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/python",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Syntax", "Loops", "Functions"],
                "description": "Specialization covering core Python including syntax, loops, and functions"
            },
            {
                "title": "Complete Python Bootcamp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/complete-python-bootcamp/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["OOP", "Automation"],
                "description": "Course covering practical Python including OOP and automation"
            },
            {
                "title": "Python Scripting Essentials",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com/courses",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Scripting", "Automation"],
                "description": "Course covering Python scripting including automation and scripting patterns"
            },
            {
                "title": "Real Python Tutorials (Course Bundles)",
                "provider": "RealPython",
                "url": "https://realpython.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["APIs", "Async", "Data"],
                "description": "Course covering applied Python including APIs, async programming, and data handling"
            },
            {
                "title": "Applied Python for Systems",
                "provider": "Linux Academy",
                "url": "https://acloudguru.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Scripts", "CLI Tools"],
                "description": "Course covering Python for DevOps including scripts and CLI tools"
            },
            {
                "title": "Python OOP Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/python-oop/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Classes", "Polymorphism"],
                "description": "Course covering OOP in Python including classes and polymorphism"
            },
            {
                "title": "Python for Automation (Selenium + Requests)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/python-automation/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Selenium", "HTTP"],
                "description": "Course covering automation including Selenium and HTTP requests"
            },
            {
                "title": "Python for Finance",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-finance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pandas", "Financial Modeling"],
                "description": "Course covering financial analytics including Pandas and financial modeling"
            },
            {
                "title": "Python for Networking Engineers",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sockets", "Netmiko"],
                "description": "Course covering network automation including sockets and netmiko"
            },
            {
                "title": "Intermediate Python (PCEP → PCAP)",
                "provider": "Python Institute",
                "url": "https://pythoninstitute.org",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Modules", "OOP"],
                "description": "Course covering Python certification including modules and OOP"
            }
        ]
        
        log.info(f"📚 Adding {len(python_courses)} Python courses to knowledge base...")
        return self.batch_add_courses(python_courses)
    
    def add_java_courses(self) -> bool:
        """
        Add Java programming courses to knowledge base.
        Includes 10 courses covering Java fundamentals, OOP, concurrency, Spring, and advanced topics.
        
        Returns:
            True if successful, False otherwise
        """
        java_courses = [
            {
                "title": "Java Programming Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/java-the-complete-java-developer-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["OOP", "Collections"],
                "description": "Course covering core Java including OOP and collections"
            },
            {
                "title": "Java Programming Basics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/java-programming",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Methods", "Logic"],
                "description": "Course covering Java fundamentals including methods and logic"
            },
            {
                "title": "Java Multithreading & Concurrency",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/java-multithreading/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Threads", "Executors"],
                "description": "Course covering concurrency including threads and executors"
            },
            {
                "title": "Spring Framework Basics",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spring Boot"],
                "description": "Course covering Java backend including Spring Boot framework"
            },
            {
                "title": "JavaFX GUI Programming",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/javafx/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GUI", "Event Handling"],
                "description": "Course covering desktop UI including GUI and event handling"
            },
            {
                "title": "Advanced Java Programming",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["JVM", "Memory"],
                "description": "Course covering Java internals including JVM and memory management"
            },
            {
                "title": "Java Microservices with Spring Cloud",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/spring-cloud-microservices/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["API Gateway", "Discovery"],
                "description": "Course covering microservices including API gateway and service discovery"
            },
            {
                "title": "Java Data Structures",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/java-data-structures",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Trees", "Graphs"],
                "description": "Course covering data structures and algorithms including trees and graphs"
            },
            {
                "title": "JDBC & SQL in Java",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/java-jdbc",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SQL", "ORM"],
                "description": "Course covering databases including SQL and ORM"
            },
            {
                "title": "Java Security Essentials",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Encryption", "Auth"],
                "description": "Course covering secure Java applications including encryption and authentication"
            }
        ]
        
        log.info(f"📚 Adding {len(java_courses)} Java courses to knowledge base...")
        return self.batch_add_courses(java_courses)
    
    def add_c_cpp_courses(self) -> bool:
        """
        Add C/C++ programming courses to knowledge base.
        Includes 10 courses covering C/C++ fundamentals, systems programming, advanced concepts, and specialized applications.
        
        Returns:
            True if successful, False otherwise
        """
        c_cpp_courses = [
            {
                "title": "C Programming for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/c-programming-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pointers", "Arrays"],
                "description": "Course covering C basics including pointers and arrays"
            },
            {
                "title": "C++ Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/beginning-c-plus-plus-programming/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["OOP", "STL"],
                "description": "Course covering C++ OOP including Standard Template Library"
            },
            {
                "title": "Advanced C++ Programming",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/advanced-c-programming/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Multithreading", "Templates"],
                "description": "Course covering advanced concepts including multithreading and templates"
            },
            {
                "title": "C Systems Programming",
                "provider": "Linux Academy",
                "url": "https://acloudguru.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Memory", "I/O"],
                "description": "Course covering systems programming including memory management and I/O"
            },
            {
                "title": "C++ Data Structures",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/c-plus-plus-data-structures",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["STL", "Recursion"],
                "description": "Course covering data structures with C++ including STL and recursion"
            },
            {
                "title": "Embedded C Programming",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/embedded-c/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hardware", "Microcontrollers"],
                "description": "Course covering embedded systems including hardware and microcontrollers"
            },
            {
                "title": "Modern C++ (C++11/14/17)",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Smart Pointers", "Lambdas"],
                "description": "Course covering modern features including smart pointers and lambdas"
            },
            {
                "title": "Game Programming in C++",
                "provider": "Udacity",
                "url": "https://www.udacity.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engines", "SDL"],
                "description": "Course covering game development including engines and SDL"
            },
            {
                "title": "Memory Management in C++",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cpp-memory/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Allocation", "RAII"],
                "description": "Course covering memory management including allocation and RAII"
            },
            {
                "title": "C++ Concurrency",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Threads", "Mutexes"],
                "description": "Course covering concurrency including threads and mutexes"
            }
        ]
        
        log.info(f"📚 Adding {len(c_cpp_courses)} C/C++ courses to knowledge base...")
        return self.batch_add_courses(c_cpp_courses)
    
    def add_javascript_courses(self) -> bool:
        """
        Add JavaScript programming courses to knowledge base.
        Includes 10 courses covering JavaScript fundamentals, advanced concepts, async programming, testing, and Node.js.
        
        Returns:
            True if successful, False otherwise
        """
        javascript_courses = [
            {
                "title": "Modern JavaScript from Scratch",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/modern-javascript/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DOM", "ES6"],
                "description": "Course covering JS basics including DOM manipulation and ES6 features"
            },
            {
                "title": "JavaScript Algorithms and Data Structures",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/js-algorithms-and-data-structures/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Recursion", "Graphs"],
                "description": "Course covering data structures in JS including recursion and graphs"
            },
            {
                "title": "Advanced JavaScript Concepts",
                "provider": "ZTM Academy",
                "url": "https://zerotomastery.io/courses/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Closures", "Event Loop"],
                "description": "Course covering JS internals including closures and event loop"
            },
            {
                "title": "JavaScript Testing with Jest",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/jest-testing/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mocks", "Snapshots"],
                "description": "Course covering testing including mocks and snapshots"
            },
            {
                "title": "Async JavaScript Mastery",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Promises", "Async/Await"],
                "description": "Course covering async code including promises and async/await"
            },
            {
                "title": "Node.js Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/nodejs",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Express", "HTTP"],
                "description": "Course covering server-side JS including Express and HTTP"
            },
            {
                "title": "JavaScript Design Patterns",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/javascript-design-patterns/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["MVC", "Factory"],
                "description": "Course covering architecture including MVC and factory patterns"
            },
            {
                "title": "Functional Programming in JavaScript",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/functional-javascript/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Immutability"],
                "description": "Course covering FP patterns including immutability"
            },
            {
                "title": "JavaScript Performance Optimization",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Profiling", "Memory"],
                "description": "Course covering performance including profiling and memory management"
            },
            {
                "title": "Testing JavaScript Applications",
                "provider": "Pluralsight",
                "url": "https://www.pluralsight.com",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mocha", "Chai"],
                "description": "Course covering testing including Mocha and Chai frameworks"
            }
        ]
        
        log.info(f"📚 Adding {len(javascript_courses)} JavaScript courses to knowledge base...")
        return self.batch_add_courses(javascript_courses)
    
    def add_product_management_courses(self) -> bool:
        """
        Add Product Management courses to knowledge base.
        Includes 10 courses covering product management fundamentals, strategy, and certifications.
        
        Returns:
            True if successful, False otherwise
        """
        product_courses = [
            {
                "title": "Digital Product Management: Modern Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/uva-darden-digital-product-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Strategy", "User Research", "Roadmapping"],
                "description": "Course covering modern fundamentals of digital product management including strategy, user research, and roadmapping"
            },
            {
                "title": "Product Management Professional Certificate",
                "provider": "Kellogg (Emeritus)",
                "url": "https://emeritus.org/in/professional-courses/kellogg-executive-education-product-management-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Strategy", "Leadership", "Business Acumen"],
                "description": "Professional certificate covering product management including strategy, leadership, and business acumen"
            },
            {
                "title": "MicroMasters® in Digital Product Management",
                "provider": "edX",
                "url": "https://www.edx.org/micromasters/bostonuniversity-product-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Products", "Agile", "Product Lifecycle"],
                "description": "MicroMasters program covering digital product management including agile methodologies and product lifecycle"
            },
            {
                "title": "Product Management Certification (PMC)",
                "provider": "Product School",
                "url": "https://productschool.com/certifications/product-manager-certification",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Development", "Stakeholder Management", "Metrics"],
                "description": "Certification covering product management including product development, stakeholder management, and metrics"
            },
            {
                "title": "Google Project Management (Foundations)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-project-management",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Planning", "Execution"],
                "description": "Course covering project management foundations including planning and execution"
            },
            {
                "title": "Become a Product Manager",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/become-a-product-manager-learn-the-skills-get-a-job/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Management", "Career Development", "Skills"],
                "description": "Course covering product management skills and career development"
            },
            {
                "title": "Technical Product Manager Nanodegree",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/technical-product-manager-nanodegree--nd046",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technical Products", "Engineering Collaboration", "Technical Strategy"],
                "description": "Nanodegree covering technical product management including engineering collaboration and technical strategy"
            },
            {
                "title": "Brand and Product Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/brand-product-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Branding", "Product Positioning", "Marketing"],
                "description": "Course covering brand and product management including branding, product positioning, and marketing"
            },
            {
                "title": "Professional Scrum Product Owner (PSPO)",
                "provider": "Scrum.org",
                "url": "https://www.scrum.org/professional-scrum-product-owner-certifications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Scrum", "Product Ownership", "Agile"],
                "description": "Certification covering professional scrum product ownership including agile methodologies"
            },
            {
                "title": "Product Strategy",
                "provider": "Reforge",
                "url": "https://www.reforge.com/product-strategy",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Strategy", "Growth", "Business Model"],
                "description": "Course covering product strategy including growth strategies and business models"
            }
        ]
        
        log.info(f"📚 Adding {len(product_courses)} Product Management courses to knowledge base...")
        return self.batch_add_courses(product_courses)
    
    def add_project_management_courses(self) -> bool:
        """
        Add Project Management courses to knowledge base.
        Includes 10 courses covering project management methodologies, certifications, and tools.
        
        Returns:
            True if successful, False otherwise
        """
        project_courses = [
            {
                "title": "Google Project Management: Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Planning", "Risk Management"],
                "description": "Professional certificate covering project management including planning and risk management"
            },
            {
                "title": "PMP® Certification Training",
                "provider": "PMI",
                "url": "https://www.pmi.org/certifications/project-management-pmp",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PMP", "Project Management", "Certification"],
                "description": "Training course for PMP certification covering project management best practices"
            },
            {
                "title": "PRINCE2® Foundation & Practitioner",
                "provider": "Axelos",
                "url": "https://www.axelos.com/certifications/prince2/prince2-foundation-and-practitioner",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PRINCE2", "Methodology", "Project Framework"],
                "description": "Certification covering PRINCE2 methodology and project framework"
            },
            {
                "title": "Agile Project Management",
                "provider": "edX",
                "url": "https://www.edx.org/professional-certificate/umd-usmx-agile-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Agile", "Scrum", "Kanban"],
                "description": "Professional certificate covering agile project management including Scrum and Kanban"
            },
            {
                "title": "Certified ScrumMaster® (CSM)",
                "provider": "Scrum Alliance",
                "url": "https://www.scrumalliance.org/get-certified/scrum-master-track/certified-scrummaster",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Scrum", "Agile", "Team Facilitation"],
                "description": "Certification covering Scrum Master skills including agile methodologies and team facilitation"
            },
            {
                "title": "Project Management Fundamentals",
                "provider": "PMI",
                "url": "https://www.pmi.org/learning/courses/project-management-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Basics", "Fundamentals"],
                "description": "Course covering project management fundamentals and basics"
            },
            {
                "title": "Introduction to Project Management",
                "provider": "edX",
                "url": "https://www.edx.org/course/introduction-to-project-management",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Introduction", "Basics"],
                "description": "Course introducing project management concepts and basics"
            },
            {
                "title": "Engineering Project Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/engineering-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Projects", "Technical Management", "Planning"],
                "description": "Specialization covering engineering project management including technical management and planning"
            },
            {
                "title": "Jira Fundamentals Badge",
                "provider": "Atlassian",
                "url": "https://university.atlassian.com/student/path/815463-jira-fundamentals-badge",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Jira", "Project Tracking", "Agile Tools"],
                "description": "Course covering Jira fundamentals including project tracking and agile tools"
            },
            {
                "title": "The Project Management Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-project-management-course-beginner-to-project-manager/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Career Development", "Skills"],
                "description": "Course covering project management from beginner to project manager level"
            }
        ]
        
        log.info(f"📚 Adding {len(project_courses)} Project Management courses to knowledge base...")
        return self.batch_add_courses(project_courses)
    
    def add_digital_marketing_courses(self) -> bool:
        """
        Add Digital Marketing courses to knowledge base.
        Includes 10 courses covering digital marketing, social media, SEO, and analytics.
        
        Returns:
            True if successful, False otherwise
        """
        marketing_courses = [
            {
                "title": "Google Digital Marketing & E-commerce",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-digital-marketing-ecommerce",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Marketing", "E-commerce", "Google Ads"],
                "description": "Professional certificate covering digital marketing and e-commerce including Google Ads"
            },
            {
                "title": "Meta Social Media Marketing Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/facebook-social-media-marketing",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Social Media Marketing", "Facebook", "Instagram"],
                "description": "Certificate covering social media marketing including Facebook and Instagram"
            },
            {
                "title": "Digital Marketing Specialist",
                "provider": "Simplilearn",
                "url": "https://www.simplilearn.com/advanced-digital-marketing-certification-training-course",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Marketing", "SEO", "Content Marketing"],
                "description": "Specialist certification covering advanced digital marketing including SEO and content marketing"
            },
            {
                "title": "Strategic Social Media Marketing",
                "provider": "edX",
                "url": "https://www.edx.org/course/strategic-social-media-marketing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Social Media Strategy", "Content Strategy", "Engagement"],
                "description": "Course covering strategic social media marketing including content strategy and engagement"
            },
            {
                "title": "HubSpot Content Marketing",
                "provider": "HubSpot",
                "url": "https://academy.hubspot.com/courses/content-marketing",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Content Marketing", "Inbound Marketing", "HubSpot"],
                "description": "Course covering content marketing including inbound marketing and HubSpot tools"
            },
            {
                "title": "Digital Marketing Nanodegree",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/digital-marketing-nanodegree--nd018",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Marketing", "Analytics", "Campaign Management"],
                "description": "Nanodegree covering digital marketing including analytics and campaign management"
            },
            {
                "title": "Advanced Search Engine Optimization (SEO)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/seo-strategies",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SEO", "Keyword Research", "Link Building"],
                "description": "Course covering advanced SEO strategies including keyword research and link building"
            },
            {
                "title": "Postgraduate Diploma in Digital Marketing",
                "provider": "DMI",
                "url": "https://digitalmarketinginstitute.com/students/courses/postgraduate-diploma-in-digital-marketing",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Marketing", "Strategy", "Advanced Techniques"],
                "description": "Postgraduate diploma covering digital marketing including strategy and advanced techniques"
            },
            {
                "title": "Marketing Analytics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/marketing-analytics-with-facebook",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Marketing Analytics", "Facebook Analytics", "Data Analysis"],
                "description": "Course covering marketing analytics including Facebook analytics and data analysis"
            },
            {
                "title": "The Complete Digital Marketing Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/learn-digital-marketing-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Marketing", "SEO", "Social Media", "Email Marketing"],
                "description": "Comprehensive course covering digital marketing including SEO, social media, and email marketing"
            }
        ]
        
        log.info(f"📚 Adding {len(marketing_courses)} Digital Marketing courses to knowledge base...")
        return self.batch_add_courses(marketing_courses)
    
    def add_business_analytics_courses(self) -> bool:
        """
        Add Business Analytics courses to knowledge base.
        Includes 10 courses covering business analytics, data analysis, and decision-making.
        
        Returns:
            True if successful, False otherwise
        """
        analytics_courses = [
            {
                "title": "Business Analytics Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/business-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Analytics", "Data Analysis", "Decision Making"],
                "description": "Specialization covering business analytics including data analysis and decision making"
            },
            {
                "title": "Business Analytics Nanodegree",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/business-analytics-nanodegree--nd098",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Analytics", "SQL", "Excel", "Tableau"],
                "description": "Nanodegree covering business analytics including SQL, Excel, and Tableau"
            },
            {
                "title": "MicroMasters® in Statistics and Data Science",
                "provider": "edX",
                "url": "https://micromasters.mit.edu/ds/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statistics", "Data Science", "Analytics"],
                "description": "MicroMasters program covering statistics and data science for business analytics"
            },
            {
                "title": "Harvard Business Analytics Program",
                "provider": "Harvard Online",
                "url": "https://analytics.hbs.edu/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Analytics", "Strategy", "Leadership"],
                "description": "Program covering business analytics including strategy and leadership"
            },
            {
                "title": "Excel to MySQL: Analytic Techniques",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/excel-mysql",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Excel", "MySQL", "Data Analysis", "SQL"],
                "description": "Specialization covering analytics techniques from Excel to MySQL including SQL"
            },
            {
                "title": "Imperial Business Analytics",
                "provider": "Imperial College",
                "url": "https://www.imperial.ac.uk/business-school/executive-education/technology-analytics-data-science/business-analytics-data-decisions/online/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Analytics", "Data Decisions", "Strategy"],
                "description": "Program covering business analytics and data-driven decision making"
            },
            {
                "title": "Data Analysis for Management",
                "provider": "LSE",
                "url": "https://www.lse.ac.uk/study-at-lse/online-learning/courses/data-analysis-for-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Analysis", "Management", "Statistics"],
                "description": "Course covering data analysis for management including statistics"
            },
            {
                "title": "Tableau 2024 A-Z",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/tableau10/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tableau", "Data Visualization", "Dashboards"],
                "description": "Course covering Tableau for data visualization and dashboard creation"
            },
            {
                "title": "SQL for Business Analysts",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/courses/sql-for-business-analysts",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SQL", "Business Analysis", "Data Querying"],
                "description": "Course covering SQL for business analysts including data querying"
            },
            {
                "title": "Marketing Analytics: Data Tools",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-marketing-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Marketing Analytics", "Data Tools", "Customer Analysis"],
                "description": "Course covering marketing analytics including data tools and customer analysis"
            }
        ]
        
        log.info(f"📚 Adding {len(analytics_courses)} Business Analytics courses to knowledge base...")
        return self.batch_add_courses(analytics_courses)
    
    def add_entrepreneurship_courses(self) -> bool:
        """
        Add Entrepreneurship courses to knowledge base.
        Includes 10 courses covering entrepreneurship, startups, innovation, and business planning.
        
        Returns:
            True if successful, False otherwise
        """
        entrepreneurship_courses = [
            {
                "title": "Entrepreneurship Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/wharton-entrepreneurship",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Entrepreneurship", "Business Planning", "Innovation"],
                "description": "Specialization covering entrepreneurship including business planning and innovation"
            },
            {
                "title": "Startup School",
                "provider": "Y Combinator",
                "url": "https://www.startupschool.org/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Startups", "YC Methodology", "Founder Skills"],
                "description": "Course covering startup fundamentals including Y Combinator methodology and founder skills"
            },
            {
                "title": "How to Build a Startup",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/how-to-build-a-startup--ep245",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Startup Building", "Lean Startup", "MVP"],
                "description": "Course covering how to build a startup including lean startup methodology and MVP"
            },
            {
                "title": "Entrepreneurship in Emerging Economies",
                "provider": "edX",
                "url": "https://www.edx.org/course/entrepreneurship-in-emerging-economies",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Emerging Markets", "Social Entrepreneurship", "Innovation"],
                "description": "Course covering entrepreneurship in emerging economies including social entrepreneurship"
            },
            {
                "title": "Zero to One: How to Build the Future",
                "provider": "MasterClass",
                "url": "https://www.masterclass.com/classes/peter-thiel-teaches-entrepreneurship",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Innovation", "Strategy", "Monopoly Theory"],
                "description": "Course covering entrepreneurship and innovation including strategy and monopoly theory"
            },
            {
                "title": "Technology Entrepreneurship",
                "provider": "edX",
                "url": "https://www.edx.org/course/technology-entrepreneurship-lab-to-market",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tech Startups", "Product Development", "Market Entry"],
                "description": "Course covering technology entrepreneurship from lab to market including product development"
            },
            {
                "title": "Innovation and Entrepreneurship",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/innovation-entrepreneurship",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Innovation", "Entrepreneurship", "Business Model"],
                "description": "Course covering innovation and entrepreneurship including business model development"
            },
            {
                "title": "Becoming an Entrepreneur",
                "provider": "edX",
                "url": "https://www.edx.org/course/becoming-an-entrepreneur",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Entrepreneurship", "Career Development", "Skills"],
                "description": "Course covering becoming an entrepreneur including career development and skills"
            },
            {
                "title": "Essentials of Entrepreneurship",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/entrepreneurship-thinking-action",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Entrepreneurship", "Thinking", "Action"],
                "description": "Course covering essentials of entrepreneurship including entrepreneurial thinking and action"
            },
            {
                "title": "The Complete Business Plan Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-business-plan-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Plan", "Financial Projections", "Pitch Deck"],
                "description": "Course covering complete business plan creation including financial projections and pitch deck"
            }
        ]
        
        log.info(f"📚 Adding {len(entrepreneurship_courses)} Entrepreneurship courses to knowledge base...")
        return self.batch_add_courses(entrepreneurship_courses)
    
    def add_finance_fintech_courses(self) -> bool:
        """
        Add Finance and FinTech courses to knowledge base.
        Includes 10 courses covering finance, FinTech, DeFi, and financial analysis.
        
        Returns:
            True if successful, False otherwise
        """
        finance_courses = [
            {
                "title": "FinTech: Foundations & Payments",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/fintech",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FinTech", "Payments", "Blockchain"],
                "description": "Course covering FinTech foundations and payments including blockchain technology"
            },
            {
                "title": "CFA® Program (Level I)",
                "provider": "CFA Institute",
                "url": "https://www.cfainstitute.org/en/programs/cfa",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CFA", "Financial Analysis", "Investment Management"],
                "description": "CFA Level I program covering financial analysis and investment management"
            },
            {
                "title": "Financial Markets",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/financial-markets-global",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Markets", "Trading", "Investments"],
                "description": "Course covering global financial markets including trading and investments"
            },
            {
                "title": "FinTech Professional Certificate",
                "provider": "edX",
                "url": "https://www.edx.org/professional-certificate/hkux-fintech",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FinTech", "Digital Finance", "Innovation"],
                "description": "Professional certificate covering FinTech including digital finance and innovation"
            },
            {
                "title": "Oxford Fintech Programme",
                "provider": "Oxford",
                "url": "https://www.getsmarter.com/products/oxford-university-fintech-programme",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FinTech", "Strategy", "Leadership"],
                "description": "Program covering FinTech including strategy and leadership"
            },
            {
                "title": "Decentralized Finance (DeFi) Infrastructure",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/decentralized-finance-infrastructure-duke",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DeFi", "Blockchain", "Smart Contracts"],
                "description": "Course covering decentralized finance infrastructure including blockchain and smart contracts"
            },
            {
                "title": "Financial Modeling & Valuation (FMVA)",
                "provider": "CFI",
                "url": "https://corporatefinanceinstitute.com/certifications/financial-modeling-valuation-analyst-fmva-program/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Modeling", "Valuation", "Excel"],
                "description": "Certification covering financial modeling and valuation including Excel"
            },
            {
                "title": "Introduction to FinTech",
                "provider": "edX",
                "url": "https://www.edx.org/course/introduction-to-fintech",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FinTech", "Introduction", "Basics"],
                "description": "Course introducing FinTech concepts and basics"
            },
            {
                "title": "Python and Statistics for Financial Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-statistics-financial-analysis",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "Statistics", "Financial Analysis"],
                "description": "Course covering Python and statistics for financial analysis"
            },
            {
                "title": "Private Equity and Venture Capital",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/private-equity",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Private Equity", "Venture Capital", "Investments"],
                "description": "Course covering private equity and venture capital investments"
            }
        ]
        
        log.info(f"📚 Adding {len(finance_courses)} Finance/FinTech courses to knowledge base...")
        return self.batch_add_courses(finance_courses)
    
    def add_hr_analytics_courses(self) -> bool:
        """
        Add HR Analytics courses to knowledge base.
        Includes 10 courses covering people analytics, HR data analysis, and talent management.
        
        Returns:
            True if successful, False otherwise
        """
        hr_courses = [
            {
                "title": "People Analytics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-people-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["People Analytics", "HR Data", "Workforce Analysis"],
                "description": "Course covering people analytics including HR data and workforce analysis"
            },
            {
                "title": "HR Analytics Certificate Program",
                "provider": "AIHR",
                "url": "https://www.aihr.com/courses/hr-analytics-certificate/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HR Analytics", "Data Analysis", "Metrics"],
                "description": "Certificate program covering HR analytics including data analysis and metrics"
            },
            {
                "title": "Human Resources Analytics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/human-resources-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HR Analytics", "Talent Management", "Data"],
                "description": "Course covering human resources analytics including talent management and data"
            },
            {
                "title": "People Analytics for HR",
                "provider": "HCI",
                "url": "https://www.hci.org/analytics-talent-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["People Analytics", "Talent Management", "HR Strategy"],
                "description": "Course covering people analytics for HR including talent management and strategy"
            },
            {
                "title": "Diploma in HR Analytics",
                "provider": "Alison",
                "url": "https://alison.com/course/diploma-in-hr-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HR Analytics", "Data Analysis", "Reporting"],
                "description": "Diploma covering HR analytics including data analysis and reporting"
            },
            {
                "title": "Introduction to People Analytics",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/people-analytics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["People Analytics", "Introduction", "Basics"],
                "description": "Course introducing people analytics concepts and basics"
            },
            {
                "title": "Strategic HR Analytics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/strategic-hr-analytics/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategic HR", "Analytics", "Decision Making"],
                "description": "Course covering strategic HR analytics including decision making"
            },
            {
                "title": "People Analytics: Collaborating for Success",
                "provider": "CIPD",
                "url": "https://www.cipd.co.uk/learn/training/people-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["People Analytics", "Collaboration", "HR Strategy"],
                "description": "Course covering people analytics including collaboration and HR strategy"
            },
            {
                "title": "Applied Data Science for HR",
                "provider": "Dartmouth (Tuck)",
                "url": "https://next.tuck.dartmouth.edu/programs/applied-data-science-for-hr",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Science", "HR", "Machine Learning"],
                "description": "Program covering applied data science for HR including machine learning"
            },
            {
                "title": "Wharton People Analytics Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/wharton-people-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["People Analytics", "HR Strategy", "Data Analysis"],
                "description": "Specialization covering people analytics including HR strategy and data analysis"
            }
        ]
        
        log.info(f"📚 Adding {len(hr_courses)} HR Analytics courses to knowledge base...")
        return self.batch_add_courses(hr_courses)
    
    def add_ui_ux_design_courses(self) -> bool:
        """
        Add UI/UX Design courses to knowledge base.
        Includes 10 courses covering user experience design, interface design, and design thinking.
        
        Returns:
            True if successful, False otherwise
        """
        ui_ux_courses = [
            {
                "title": "Google UX Design Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-ux-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Design", "User Research", "Prototyping", "Wireframing"],
                "description": "Professional certificate covering UX design including user research, prototyping, and wireframing"
            },
            {
                "title": "UI / UX Design Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/ui-ux-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UI Design", "UX Design", "Design Process", "User Testing"],
                "description": "Specialization covering UI/UX design including design process and user testing"
            },
            {
                "title": "User Experience Research and Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/michiganux",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Research", "User Testing", "Design Thinking", "Prototyping"],
                "description": "Specialization covering user experience research and design including design thinking and prototyping"
            },
            {
                "title": "Meta Principles of UX/UI Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/principles-of-ux-ui-design",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX/UI Principles", "Design Fundamentals", "User-Centered Design"],
                "description": "Course covering principles of UX/UI design including design fundamentals and user-centered design"
            },
            {
                "title": "Product Design (UI/UX) Master Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/product-design-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Product Design", "UI/UX", "Design Systems", "Figma"],
                "description": "Master course covering product design including UI/UX, design systems, and Figma"
            },
            {
                "title": "Complete Web & Mobile Designer (UI/UX)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/complete-web-designer-mobile-designer-zero-to-mastery/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Web Design", "Mobile Design", "UI/UX", "Responsive Design"],
                "description": "Course covering complete web and mobile design including UI/UX and responsive design"
            },
            {
                "title": "Enterprise Design Thinking Practitioner",
                "provider": "IBM",
                "url": "https://www.ibm.com/design/thinking/page/badges/practitioner",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Design Thinking", "Enterprise Design", "Innovation"],
                "description": "Course covering enterprise design thinking including innovation methodologies"
            },
            {
                "title": "Digital Experience Design (DX)",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/digital-experience-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Experience", "User Journey", "Experience Design"],
                "description": "Course covering digital experience design including user journey mapping"
            },
            {
                "title": "Introduction to User Experience Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/user-experience-design",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Design", "Introduction", "Design Basics"],
                "description": "Course introducing user experience design concepts and basics"
            },
            {
                "title": "User Interface Design Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/user-interface-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UI Design", "Interface Design", "Visual Design", "Interaction Design"],
                "description": "Specialization covering user interface design including visual and interaction design"
            }
        ]
        
        log.info(f"📚 Adding {len(ui_ux_courses)} UI/UX Design courses to knowledge base...")
        return self.batch_add_courses(ui_ux_courses)
    
    def add_graphic_design_courses(self) -> bool:
        """
        Add Graphic Design courses to knowledge base.
        Includes 10 courses covering graphic design fundamentals, branding, and design tools.
        
        Returns:
            True if successful, False otherwise
        """
        graphic_design_courses = [
            {
                "title": "Graphic Design Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/graphic-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Graphic Design", "Typography", "Layout", "Color Theory"],
                "description": "Specialization covering graphic design including typography, layout, and color theory"
            },
            {
                "title": "Graphic Design Elements for Non-Designers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/graphic-design-elements-non-designers",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Graphic Design", "Design Elements", "Basics"],
                "description": "Specialization covering graphic design elements for non-designers"
            },
            {
                "title": "The Complete Graphic Design Theory",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/graphic-design-theory-for-beginners-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Design Theory", "Color Theory", "Typography", "Composition"],
                "description": "Course covering complete graphic design theory including color theory, typography, and composition"
            },
            {
                "title": "Adobe Illustrator CC – Essentials Training",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adobe-illustrator-cc-essentials-training-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Adobe Illustrator", "Vector Graphics", "Illustration"],
                "description": "Course covering Adobe Illustrator CC essentials including vector graphics and illustration"
            },
            {
                "title": "Graphic Design Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/graphic-design-masterclass-everything-you-need-to-know/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Graphic Design", "Design Principles", "Software", "Projects"],
                "description": "Masterclass covering comprehensive graphic design including design principles, software, and projects"
            },
            {
                "title": "Fundamentals of Graphic Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/fundamentals-of-graphic-design",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Graphic Design", "Fundamentals", "Design Basics"],
                "description": "Course covering fundamentals of graphic design and design basics"
            },
            {
                "title": "Logo Design Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/logo-design-fundamentals/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Logo Design", "Branding", "Identity Design"],
                "description": "Course covering logo design fundamentals including branding and identity design"
            },
            {
                "title": "Branding and Identity",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/brand-identity-strategy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Branding", "Identity Design", "Brand Strategy"],
                "description": "Course covering branding and identity including brand strategy"
            },
            {
                "title": "Graphic Design Basics",
                "provider": "Canva",
                "url": "https://www.canva.com/designschool/courses/graphic-design-basics/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Graphic Design", "Canva", "Design Basics"],
                "description": "Course covering graphic design basics using Canva"
            },
            {
                "title": "Adobe Photoshop CC: A Beginner's Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adobe-photoshop-cc-beginners-guide/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Adobe Photoshop", "Photo Editing", "Digital Design"],
                "description": "Course covering Adobe Photoshop CC for beginners including photo editing and digital design"
            }
        ]
        
        log.info(f"📚 Adding {len(graphic_design_courses)} Graphic Design courses to knowledge base...")
        return self.batch_add_courses(graphic_design_courses)
    
    def add_animation_courses(self) -> bool:
        """
        Add Animation courses to knowledge base.
        Includes 10 courses covering 2D/3D animation, motion graphics, and animation principles.
        
        Returns:
            True if successful, False otherwise
        """
        animation_courses = [
            {
                "title": "Pixar in a Box",
                "provider": "Khan Academy",
                "url": "https://www.khanacademy.org/computing/pixar",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Animation", "Storytelling", "3D Animation"],
                "description": "Free course covering animation and storytelling from Pixar including 3D animation concepts"
            },
            {
                "title": "3D Animation (Maya & Blender)",
                "provider": "CG Spectrum",
                "url": "https://www.cgspectrum.com/courses/3d-animation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D Animation", "Maya", "Blender", "Character Animation"],
                "description": "Course covering 3D animation using Maya and Blender including character animation"
            },
            {
                "title": "The 12 Principles of Animation",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-12-principles-of-animation/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Animation Principles", "2D Animation", "Fundamentals"],
                "description": "Course covering the 12 principles of animation including 2D animation fundamentals"
            },
            {
                "title": "Character Design for Video Games",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/character-design-video-games",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Character Design", "Game Design", "Concept Art"],
                "description": "Course covering character design for video games including concept art"
            },
            {
                "title": "Interactive Computer Graphics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/interactive-computer-graphics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Graphics", "3D Graphics", "Rendering"],
                "description": "Course covering interactive computer graphics including 3D graphics and rendering"
            },
            {
                "title": "After Effects CC: The Complete Motion Graphics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/after-effects-cc/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["After Effects", "Motion Graphics", "VFX"],
                "description": "Course covering After Effects CC for motion graphics and VFX"
            },
            {
                "title": "2D Animation for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/animation-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["2D Animation", "Animation Basics", "Frame-by-Frame"],
                "description": "Course covering 2D animation for beginners including frame-by-frame animation"
            },
            {
                "title": "Animation Bootcamp",
                "provider": "School of Motion",
                "url": "https://www.schoolofmotion.com/animation-bootcamp",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Animation", "Motion Design", "After Effects"],
                "description": "Bootcamp covering animation including motion design and After Effects"
            },
            {
                "title": "Learn to Animate: Classical 2D Animation",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/learn-to-animate/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["2D Animation", "Classical Animation", "Traditional Animation"],
                "description": "Course covering classical 2D animation including traditional animation techniques"
            },
            {
                "title": "Blender 2.8 Essential Training",
                "provider": "LinkedIn",
                "url": "https://www.linkedin.com/learning/blender-2-8-essential-training",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Blender", "3D Animation", "Modeling"],
                "description": "Course covering Blender 2.8 essentials including 3D animation and modeling"
            }
        ]
        
        log.info(f"📚 Adding {len(animation_courses)} Animation courses to knowledge base...")
        return self.batch_add_courses(animation_courses)
    
    def add_game_development_courses(self) -> bool:
        """
        Add Game Development courses to knowledge base.
        Includes 10 courses covering game design, Unity, Unreal Engine, and game programming.
        
        Returns:
            True if successful, False otherwise
        """
        game_dev_courses = [
            {
                "title": "CS50's Introduction to Game Development",
                "provider": "edX",
                "url": "https://www.edx.org/course/cs50s-introduction-to-game-development",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Development", "Game Design", "Programming"],
                "description": "Course introducing game development including game design and programming"
            },
            {
                "title": "Game Design and Development with Unity",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/game-design-and-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unity", "Game Design", "Game Development", "C#"],
                "description": "Specialization covering game design and development with Unity including C# programming"
            },
            {
                "title": "C# Programming for Unity Game Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/programming-unity-game-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["C#", "Unity", "Game Programming", "Scripting"],
                "description": "Specialization covering C# programming for Unity game development including scripting"
            },
            {
                "title": "Unreal Engine 5 C++ Developer",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/unrealcourse/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unreal Engine", "C++", "Game Development", "Blueprints"],
                "description": "Course covering Unreal Engine 5 C++ development including blueprints"
            },
            {
                "title": "Complete C# Unity Game Developer 2D",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/unitycourse/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unity", "C#", "2D Games", "Game Development"],
                "description": "Course covering complete C# Unity game development for 2D games"
            },
            {
                "title": "Introduction to Game Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/game-design",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Design", "Game Mechanics", "Level Design"],
                "description": "Course introducing game design including game mechanics and level design"
            },
            {
                "title": "Unreal Engine 5: The Complete Beginner's Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/unreal-engine-5-the-complete-beginners-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unreal Engine 5", "Game Development", "Blueprints"],
                "description": "Course covering Unreal Engine 5 for beginners including blueprints"
            },
            {
                "title": "C++ Programming for Unreal Game Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/cplusplus-unreal-game-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["C++", "Unreal Engine", "Game Programming"],
                "description": "Specialization covering C++ programming for Unreal game development"
            }
        ]
        
        log.info(f"📚 Adding {len(game_dev_courses)} Game Development courses to knowledge base...")
        return self.batch_add_courses(game_dev_courses)
    
    def add_video_editing_courses(self) -> bool:
        """
        Add Video Editing courses to knowledge base.
        Includes 10 courses covering video editing, post-production, and video storytelling.
        
        Returns:
            True if successful, False otherwise
        """
        video_editing_courses = [
            {
                "title": "Adobe Premiere Pro CC: Video Editing",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adobe-premiere-pro-video-editing/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Premiere Pro", "Video Editing", "Post-Production"],
                "description": "Course covering Adobe Premiere Pro CC for video editing and post-production"
            },
            {
                "title": "The Art of Visual Storytelling",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/visual-storytelling",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Visual Storytelling", "Narrative", "Cinematography"],
                "description": "Course covering the art of visual storytelling including narrative and cinematography"
            },
            {
                "title": "DaVinci Resolve 18: Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/davinci-resolve-15-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DaVinci Resolve", "Color Grading", "Video Editing"],
                "description": "Course covering DaVinci Resolve 18 including color grading and video editing"
            },
            {
                "title": "Mastering Final Cut Pro",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mastering-final-cut-pro",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Final Cut Pro", "Video Editing", "Post-Production"],
                "description": "Course covering mastering Final Cut Pro for video editing and post-production"
            },
            {
                "title": "Video Editing with Adobe Premiere Pro",
                "provider": "Skillshare",
                "url": "https://www.skillshare.com/classes/Video-Editing-with-Adobe-Premiere-Pro-for-Beginners/1382464734",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Premiere Pro", "Video Editing", "Beginners"],
                "description": "Course covering video editing with Adobe Premiere Pro for beginners"
            },
            {
                "title": "Smartphone Filmmaking for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/smartphone-filmmaking-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Filmmaking", "Mobile Video", "Video Production"],
                "description": "Course covering smartphone filmmaking for beginners including mobile video production"
            },
            {
                "title": "Inside The Edit: The Edit Course",
                "provider": "Inside The Edit",
                "url": "https://www.insidetheedit.com/the-course/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Video Editing", "Editing Techniques", "Post-Production"],
                "description": "Course covering advanced video editing techniques and post-production"
            },
            {
                "title": "Learning DaVinci Resolve 16",
                "provider": "LinkedIn",
                "url": "https://www.linkedin.com/learning/learning-davinci-resolve-16",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["DaVinci Resolve", "Video Editing", "Color Correction"],
                "description": "Course covering learning DaVinci Resolve 16 including video editing and color correction"
            },
            {
                "title": "Adobe Premiere Pro CC Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adobe-premiere-pro-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Premiere Pro", "Advanced Editing", "Workflow"],
                "description": "Masterclass covering Adobe Premiere Pro CC including advanced editing and workflow"
            },
            {
                "title": "Video Production Essentials",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/video-production-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Video Production", "Filmmaking", "Video Basics"],
                "description": "Course covering video production essentials including filmmaking and video basics"
            }
        ]
        
        log.info(f"📚 Adding {len(video_editing_courses)} Video Editing courses to knowledge base...")
        return self.batch_add_courses(video_editing_courses)
    
    def add_3d_modeling_courses(self) -> bool:
        """
        Add 3D Modeling courses to knowledge base.
        Includes 10 courses covering 3D modeling, Blender, Maya, and 3D design.
        
        Returns:
            True if successful, False otherwise
        """
        modeling_courses = [
            {
                "title": "Blender 3D: From Zero to Hero",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/blender-3d-from-zero-to-hero/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Blender", "3D Modeling", "3D Design"],
                "description": "Course covering Blender 3D from zero to hero including 3D modeling and design"
            },
            {
                "title": "3D CAD Fundamental",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/3d-cad-fundamental",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D CAD", "CAD Modeling", "Engineering Design"],
                "description": "Course covering 3D CAD fundamentals including CAD modeling and engineering design"
            },
            {
                "title": "Complete Blender Creator",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/blendertutorial/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Blender", "3D Modeling", "Texturing", "Rendering"],
                "description": "Course covering complete Blender creator including 3D modeling, texturing, and rendering"
            },
            {
                "title": "3D Modeling for Beginners",
                "provider": "Domestika",
                "url": "https://www.domestika.org/en/courses/2873-3d-modeling-for-beginners-with-nomad-sculpt",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D Modeling", "Sculpting", "Digital Sculpting"],
                "description": "Course covering 3D modeling for beginners including sculpting and digital sculpting"
            },
            {
                "title": "Introduction to 3D Modeling",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intro-3d-modeling",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D Modeling", "Introduction", "Basics"],
                "description": "Course introducing 3D modeling concepts and basics"
            },
            {
                "title": "Maya for Beginners: Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/maya-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Maya", "3D Modeling", "Animation"],
                "description": "Course covering Maya for beginners including 3D modeling and animation"
            },
            {
                "title": "3D Printing Applications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/3d-printing-applications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D Printing", "Additive Manufacturing", "Prototyping"],
                "description": "Course covering 3D printing applications including additive manufacturing and prototyping"
            },
            {
                "title": "ZBrush: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/zbrush-the-complete-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ZBrush", "Digital Sculpting", "Character Modeling"],
                "description": "Course covering ZBrush including digital sculpting and character modeling"
            },
            {
                "title": "SketchUp: modelling simple 3D objects",
                "provider": "Coursera",
                "url": "https://www.coursera.org/projects/sketchup-modelling-simple-3d-objects",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SketchUp", "3D Modeling", "Architectural Design"],
                "description": "Course covering SketchUp for modeling simple 3D objects including architectural design"
            },
            {
                "title": "Blender for Beginners",
                "provider": "Skillshare",
                "url": "https://www.skillshare.com/classes/Blender-3D-Your-First-3D-Character/1339326871",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Blender", "3D Character", "Modeling"],
                "description": "Course covering Blender for beginners including 3D character modeling"
            }
        ]
        
        log.info(f"📚 Adding {len(modeling_courses)} 3D Modeling courses to knowledge base...")
        return self.batch_add_courses(modeling_courses)
    
    def add_iot_courses(self) -> bool:
        """
        Add Internet of Things (IoT) courses to knowledge base.
        Includes 10 courses covering IoT fundamentals, embedded systems, wireless computing, and cloud integration.
        
        Returns:
            True if successful, False otherwise
        """
        iot_courses = [
            {
                "title": "Introduction to the Internet of Things and Embedded Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/iot",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT", "Embedded Systems", "Sensors", "Actuators"],
                "description": "Course introducing Internet of Things and embedded systems including sensors and actuators"
            },
            {
                "title": "An Introduction to Programming the Internet of Things (IOT)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/iot",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT Programming", "Embedded Programming", "Arduino", "Raspberry Pi"],
                "description": "Specialization covering programming the Internet of Things including Arduino and Raspberry Pi"
            },
            {
                "title": "Internet of Things (IoT) Fundamentals",
                "provider": "edX",
                "url": "https://www.edx.org/learn/iot-internet-of-things/curtin-university-internet-of-things-iot-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT Fundamentals", "Connected Devices", "IoT Architecture"],
                "description": "Course covering IoT fundamentals including connected devices and IoT architecture"
            },
            {
                "title": "Introduction to IoT",
                "provider": "Cisco Networking Academy",
                "url": "https://www.netacad.com/courses/iot/introduction-iot",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT", "Networking", "Cisco IoT"],
                "description": "Course introducing IoT concepts including networking and Cisco IoT"
            },
            {
                "title": "Industrial Internet of Things (IIoT)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/industrial-iot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IIoT", "Industrial IoT", "Manufacturing", "Automation"],
                "description": "Course covering Industrial Internet of Things including manufacturing and automation"
            },
            {
                "title": "IoT (Internet of Things) Wireless & Cloud Computing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/iot-wireless-cloud-computing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT", "Wireless Communication", "Cloud Computing", "IoT Platforms"],
                "description": "Course covering IoT wireless communication and cloud computing including IoT platforms"
            },
            {
                "title": "Developing Industrial Internet of Things",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/developing-industrial-iot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IIoT Development", "Industrial Systems", "IoT Architecture"],
                "description": "Specialization covering developing Industrial Internet of Things including industrial systems"
            },
            {
                "title": "Architecting Smart IoT Devices",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/iot-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT Architecture", "Smart Devices", "System Design"],
                "description": "Course covering architecting smart IoT devices including system design"
            },
            {
                "title": "AWS IoT: Developing and Deploying an Internet of Things",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/aws-iot-developing-and-deploying-an-internet-of-things/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AWS IoT", "Cloud IoT", "IoT Deployment", "AWS Services"],
                "description": "Course covering AWS IoT including developing and deploying Internet of Things solutions"
            },
            {
                "title": "Complete Guide to Building an IoT System",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/complete-guide-to-building-an-iot-system/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IoT Systems", "IoT Development", "End-to-End IoT"],
                "description": "Course covering complete guide to building an IoT system from end to end"
            }
        ]
        
        log.info(f"📚 Adding {len(iot_courses)} IoT courses to knowledge base...")
        return self.batch_add_courses(iot_courses)
    
    def add_embedded_systems_courses(self) -> bool:
        """
        Add Embedded Systems courses to knowledge base.
        Includes 10 courses covering embedded systems programming, microcontrollers, real-time systems, and FPGA design.
        
        Returns:
            True if successful, False otherwise
        """
        embedded_courses = [
            {
                "title": "Introduction to Embedded Systems Software",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-embedded-systems",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Embedded Systems", "Embedded Software", "C Programming"],
                "description": "Course introducing embedded systems software including C programming"
            },
            {
                "title": "Embedded Software and Hardware Architecture",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/embedded-software-hardware-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Embedded Architecture", "Hardware", "Software", "System Design"],
                "description": "Course covering embedded software and hardware architecture including system design"
            },
            {
                "title": "Development of Real-Time Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/real-time-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Real-Time Systems", "RTOS", "Scheduling", "Timing"],
                "description": "Course covering development of real-time systems including RTOS and scheduling"
            },
            {
                "title": "Mastering Microcontroller and Embedded Driver Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-microcontroller-with-peripheral-driver-development/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Microcontrollers", "Driver Development", "Peripherals", "STM32"],
                "description": "Course covering mastering microcontroller and embedded driver development including STM32"
            },
            {
                "title": "Embedded Systems Programming on ARM Cortex-M3/M4",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/embedded-system-programming-on-arm-cortex-m3m4/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ARM Cortex", "Embedded Programming", "Cortex-M3", "Cortex-M4"],
                "description": "Course covering embedded systems programming on ARM Cortex-M3/M4"
            },
            {
                "title": "FPGA Design for Embedded Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/fpga-design",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FPGA", "Hardware Design", "Verilog", "VHDL"],
                "description": "Specialization covering FPGA design for embedded systems including Verilog and VHDL"
            },
            {
                "title": "System Validation (Embedded Systems)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/system-validation-2",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["System Validation", "Testing", "Formal Methods", "Verification"],
                "description": "Course covering system validation for embedded systems including formal methods and verification"
            },
            {
                "title": "Shape the World: Introduction to Embedded Systems",
                "provider": "edX",
                "url": "https://www.edx.org/learn/embedded-systems/the-university-of-texas-at-austin-embedded-systems-shape-the-world-microcontroller-input-output",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Embedded Systems", "Microcontrollers", "Input/Output", "Basics"],
                "description": "Course introducing embedded systems including microcontrollers and input/output"
            },
            {
                "title": "Tiny Machine Learning (TinyML)",
                "provider": "edX",
                "url": "https://www.edx.org/learn/machine-learning/harvard-university-fundamentals-of-tinyml",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["TinyML", "Edge AI", "Machine Learning", "Embedded AI"],
                "description": "Course covering fundamentals of Tiny Machine Learning including edge AI and embedded AI"
            }
        ]
        
        log.info(f"📚 Adding {len(embedded_courses)} Embedded Systems courses to knowledge base...")
        return self.batch_add_courses(embedded_courses)
    
    def add_robotics_courses(self) -> bool:
        """
        Add Robotics courses to knowledge base.
        Includes 10 courses covering robotics mechanics, planning, control, ROS, and autonomous systems.
        
        Returns:
            True if successful, False otherwise
        """
        robotics_courses = [
            {
                "title": "Modern Robotics: Mechanics, Planning, and Control",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/modernrobotics",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics", "Mechanics", "Motion Planning", "Control"],
                "description": "Specialization covering modern robotics including mechanics, motion planning, and control"
            },
            {
                "title": "Robotics Software Engineer",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/robotics-software-engineer--nd209",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics Software", "ROS", "Robot Programming", "Navigation"],
                "description": "Nanodegree covering robotics software engineering including ROS and robot programming"
            },
            {
                "title": "Robotics: Aerial Robotics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/robotics-flight",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Aerial Robotics", "Drones", "Quadrotors", "Flight Control"],
                "description": "Course covering aerial robotics including drones, quadrotors, and flight control"
            },
            {
                "title": "Artificial Intelligence for Robotics",
                "provider": "Udacity",
                "url": "https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI for Robotics", "Localization", "SLAM", "Path Planning"],
                "description": "Course covering artificial intelligence for robotics including localization, SLAM, and path planning"
            },
            {
                "title": "ROS for Beginners (ROS1 & ROS2)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ros-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ROS", "ROS1", "ROS2", "Robot Operating System"],
                "description": "Course covering ROS for beginners including ROS1 and ROS2"
            },
            {
                "title": "Control of Mobile Robots",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mobile-robot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mobile Robots", "Control Systems", "Navigation", "Robotics"],
                "description": "Course covering control of mobile robots including control systems and navigation"
            },
            {
                "title": "Building a Future with Robots",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/robotics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics", "Future Technology", "Robot Applications"],
                "description": "Course covering building a future with robots including robot applications"
            },
            {
                "title": "Self-Driving Cars Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/self-driving-cars",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Self-Driving Cars", "Autonomous Vehicles", "Computer Vision", "Sensor Fusion"],
                "description": "Specialization covering self-driving cars including autonomous vehicles, computer vision, and sensor fusion"
            },
            {
                "title": "Introduction to Robotics",
                "provider": "edX",
                "url": "https://www.edx.org/learn/robotics/massachusetts-institute-of-technology-introduction-to-robotics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics", "Introduction", "Robot Basics"],
                "description": "Course introducing robotics concepts and robot basics"
            },
            {
                "title": "Become a Robotics Software Engineer",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/robotics-software-engineer/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics Software", "Career Development", "Robot Programming"],
                "description": "Course covering becoming a robotics software engineer including career development"
            }
        ]
        
        log.info(f"📚 Adding {len(robotics_courses)} Robotics courses to knowledge base...")
        return self.batch_add_courses(robotics_courses)
    
    def add_ar_vr_courses(self) -> bool:
        """
        Add AR/VR Development courses to knowledge base.
        Includes 10 courses covering augmented reality, virtual reality, mixed reality, and XR development.
        
        Returns:
            True if successful, False otherwise
        """
        ar_vr_courses = [
            {
                "title": "XR for Everybody (AR/VR/MR)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/extended-reality-for-everybody",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["XR", "AR", "VR", "MR", "Extended Reality"],
                "description": "Specialization covering XR for everybody including AR, VR, and MR"
            },
            {
                "title": "Introduction to Augmented Reality and ARCore",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ar",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Augmented Reality", "ARCore", "AR Development", "Mobile AR"],
                "description": "Course introducing augmented reality and ARCore including mobile AR development"
            },
            {
                "title": "Virtual Reality Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/virtual-reality",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Virtual Reality", "VR Development", "3D Graphics", "VR Design"],
                "description": "Specialization covering virtual reality including VR development and 3D graphics"
            },
            {
                "title": "Unity XR: How to Build AR and VR Apps",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/unity-xr-how-to-build-ar-and-vr-apps/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unity XR", "AR Apps", "VR Apps", "Unity Development"],
                "description": "Course covering Unity XR for building AR and VR apps"
            },
            {
                "title": "CS50's Introduction to Game Development",
                "provider": "edX",
                "url": "https://www.edx.org/course/cs50s-introduction-to-game-development",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Development", "VR Games", "Game Design"],
                "description": "Course introducing game development including VR games and game design"
            },
            {
                "title": "Professional Certificate in AR/VR Development",
                "provider": "edX",
                "url": "https://www.edx.org/certificates/professional-certificate/nyux-augmented-and-virtual-reality",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AR/VR Development", "Professional Certificate", "XR Development"],
                "description": "Professional certificate covering AR/VR development including XR development"
            },
            {
                "title": "Build Virtual Reality Games for Google Cardboard",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/virtual-reality-game-development/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VR Games", "Google Cardboard", "VR Development", "Mobile VR"],
                "description": "Course covering building virtual reality games for Google Cardboard including mobile VR"
            },
            {
                "title": "Complete C# Unity Game Developer 3D",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/unitycourse2/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Unity", "C#", "3D Games", "VR Development"],
                "description": "Course covering complete C# Unity game development for 3D games including VR"
            },
            {
                "title": "Meta Spark Creator AR Certification",
                "provider": "Meta (Spark AR)",
                "url": "https://www.facebook.com/business/learn/certification/exams/100-101-meta-spark-creator",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spark AR", "AR Filters", "Meta AR", "Social AR"],
                "description": "Certification covering Meta Spark Creator AR including AR filters and social AR"
            },
            {
                "title": "Creating Virtual Reality (VR) Apps",
                "provider": "edX",
                "url": "https://www.edx.org/learn/virtual-reality/university-of-california-san-diego-creating-virtual-reality-vr-apps",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VR Apps", "VR Development", "Unity", "VR Design"],
                "description": "Course covering creating virtual reality apps including Unity and VR design"
            }
        ]
        
        log.info(f"📚 Adding {len(ar_vr_courses)} AR/VR Development courses to knowledge base...")
        return self.batch_add_courses(ar_vr_courses)
    
    def add_quantum_computing_courses(self) -> bool:
        """
        Add Quantum Computing courses to knowledge base.
        Includes 10 courses covering quantum computing fundamentals, quantum algorithms, and quantum information.
        
        Returns:
            True if successful, False otherwise
        """
        quantum_courses = [
            {
                "title": "Quantum Computing for Everyone",
                "provider": "edX",
                "url": "https://www.edx.org/learn/quantum-computing/the-university-of-chicago-quantum-computing-for-everyone",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computing", "Introduction", "Quantum Basics"],
                "description": "Course covering quantum computing for everyone including quantum basics"
            },
            {
                "title": "Introduction to Quantum Computing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/quantum-computing-algorithms",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computing", "Quantum Algorithms", "Qubits"],
                "description": "Course introducing quantum computing including quantum algorithms and qubits"
            },
            {
                "title": "Quantum Mechanics for Scientists and Engineers",
                "provider": "edX",
                "url": "https://www.edx.org/learn/quantum-mechanics/stanford-university-quantum-mechanics-for-scientists-and-engineers",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Mechanics", "Physics", "Quantum Theory"],
                "description": "Course covering quantum mechanics for scientists and engineers including quantum theory"
            },
            {
                "title": "Understanding Quantum Computers",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/quantum-computers",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computers", "Understanding", "Basics"],
                "description": "Course covering understanding quantum computers and basics"
            },
            {
                "title": "The Complete Quantum Computing Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/quantum-computing-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computing", "Quantum Algorithms", "Quantum Programming"],
                "description": "Course covering complete quantum computing including quantum algorithms and programming"
            },
            {
                "title": "Quantum Computing with IBM Qiskit",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/practical-quantum-computing-ibm-qiskit",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Qiskit", "IBM Quantum", "Quantum Programming", "Quantum Circuits"],
                "description": "Course covering quantum computing with IBM Qiskit including quantum programming and circuits"
            },
            {
                "title": "Quantum Science and Engineering",
                "provider": "edX",
                "url": "https://www.edx.org/learn/quantum-physics-mechanics/harvard-university-quantum-science-and-engineering",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Science", "Quantum Engineering", "Advanced Topics"],
                "description": "Course covering quantum science and engineering including advanced topics"
            },
            {
                "title": "QC101: Quantum Computing & Quantum Physics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/qc101-quantum-computing-quantum-physics-for-beginners/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computing", "Quantum Physics", "Basics"],
                "description": "Course covering quantum computing and quantum physics for beginners"
            },
            {
                "title": "Fundamentals of Quantum Computing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/quantum-computing-fundamentals",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Computing", "Fundamentals", "Quantum Algorithms"],
                "description": "Course covering fundamentals of quantum computing including quantum algorithms"
            },
            {
                "title": "Introduction to Quantum Information",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-quantum-information",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantum Information", "Quantum Communication", "Quantum Cryptography"],
                "description": "Course introducing quantum information including quantum communication and cryptography"
            }
        ]
        
        log.info(f"📚 Adding {len(quantum_courses)} Quantum Computing courses to knowledge base...")
        return self.batch_add_courses(quantum_courses)
    
    def add_networking_courses(self) -> bool:
        """
        Add Networking (CCNA/CCNP) courses to knowledge base.
        Includes 10 courses covering networking fundamentals, CCNA certification, network security, and IT support.
        
        Returns:
            True if successful, False otherwise
        """
        networking_courses = [
            {
                "title": "Cisco CCNA (200-301) - The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ccna-complete-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CCNA", "Cisco", "Networking", "Certification"],
                "description": "Course covering Cisco CCNA (200-301) complete guide including networking and certification prep"
            },
            {
                "title": "Introduction to Networks (CCNAv7)",
                "provider": "Cisco NetAcad",
                "url": "https://www.netacad.com/courses/networking/ccna-introduction-networks",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Networking", "CCNA", "Network Fundamentals", "Cisco"],
                "description": "Course covering introduction to networks (CCNAv7) including network fundamentals"
            },
            {
                "title": "The Complete Networking Fundamentals Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/complete-networking-fundamentals-course-ccna-start/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Networking Fundamentals", "CCNA Prep", "Network Basics"],
                "description": "Course covering complete networking fundamentals including CCNA preparation"
            },
            {
                "title": "Computer Communications Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/computer-communications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Communications", "Networking", "Protocols", "Network Architecture"],
                "description": "Specialization covering computer communications including protocols and network architecture"
            },
            {
                "title": "Cisco Networking Basics Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/cisco-networking-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cisco Networking", "Networking Basics", "Cisco Technologies"],
                "description": "Specialization covering Cisco networking basics including Cisco technologies"
            },
            {
                "title": "IP Addressing and Subnetting",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ip-addressing-and-subnetting-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IP Addressing", "Subnetting", "Network Design", "CIDR"],
                "description": "Course covering IP addressing and subnetting including network design and CIDR"
            },
            {
                "title": "CCNA 200-301 Complete Video Boot Camp",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/ccna-200-301-video-boot-camp-with-chris-bryant/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CCNA", "Certification Prep", "Networking", "Video Training"],
                "description": "Video boot camp covering CCNA 200-301 complete certification preparation"
            },
            {
                "title": "Network Security",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/network-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Network Security", "Cybersecurity", "Firewalls", "VPN"],
                "description": "Specialization covering network security including cybersecurity, firewalls, and VPN"
            },
            {
                "title": "Google IT Support Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-it-support",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IT Support", "Networking", "Troubleshooting", "System Administration"],
                "description": "Professional certificate covering Google IT support including networking and troubleshooting"
            },
            {
                "title": "Introduction to Computer Networking",
                "provider": "edX",
                "url": "https://www.edx.org/learn/computer-networking/stanford-university-introduction-to-computer-networking",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Networking", "Introduction", "Network Basics"],
                "description": "Course introducing computer networking concepts and network basics"
            }
        ]
        
        log.info(f"📚 Adding {len(networking_courses)} Networking courses to knowledge base...")
        return self.batch_add_courses(networking_courses)
    
    def add_mechanical_engineering_courses(self) -> bool:
        """
        Add Mechanical Engineering courses to knowledge base.
        Includes 10 courses covering mechanics, thermodynamics, materials, and manufacturing.
        
        Returns:
            True if successful, False otherwise
        """
        mech_eng_courses = [
            {
                "title": "Introduction to Engineering Mechanics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/engineering-mechanics-statics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Mechanics", "Statics", "Forces", "Equilibrium"],
                "description": "Course introducing engineering mechanics including statics, forces, and equilibrium"
            },
            {
                "title": "Applications in Engineering Mechanics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/engineering-mechanics-applications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Mechanics", "Applications", "Problem Solving"],
                "description": "Course covering applications in engineering mechanics including problem solving"
            },
            {
                "title": "Mechanics of Materials I: Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mechanics-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mechanics of Materials", "Stress", "Strain", "Material Properties"],
                "description": "Course covering mechanics of materials fundamentals including stress, strain, and material properties"
            },
            {
                "title": "Introduction to Thermodynamics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/thermodynamics-intro",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Thermodynamics", "Energy", "Heat Transfer", "Laws of Thermodynamics"],
                "description": "Course introducing thermodynamics including energy, heat transfer, and laws of thermodynamics"
            },
            {
                "title": "Modern Robotics: Mechanics, Planning, and Control",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/modernrobotics",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics", "Mechanics", "Motion Planning", "Control"],
                "description": "Specialization covering modern robotics including mechanics, motion planning, and control"
            },
            {
                "title": "Engineering Systems in Motion: Dynamics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/engineering-systems-in-motion",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Dynamics", "Kinematics", "Kinetics", "Motion Analysis"],
                "description": "Course covering engineering systems in motion including dynamics, kinematics, and kinetics"
            },
            {
                "title": "Material Behavior",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/material-behavior",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Material Science", "Material Behavior", "Properties", "Testing"],
                "description": "Course covering material behavior including material science, properties, and testing"
            },
            {
                "title": "Finite Element Method (FEM) for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/finite-element-method",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Finite Element Method", "FEM", "Numerical Analysis", "Simulation"],
                "description": "Course covering finite element method for engineers including numerical analysis and simulation"
            },
            {
                "title": "Machine Design Part I",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/machine-design1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Machine Design", "Mechanical Design", "Components", "Design Principles"],
                "description": "Course covering machine design including mechanical design, components, and design principles"
            },
            {
                "title": "Advanced Manufacturing Enterprise",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/advanced-manufacturing-enterprise",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Manufacturing", "Enterprise Systems", "Production", "Operations"],
                "description": "Course covering advanced manufacturing enterprise including production and operations"
            }
        ]
        
        log.info(f"📚 Adding {len(mech_eng_courses)} Mechanical Engineering courses to knowledge base...")
        return self.batch_add_courses(mech_eng_courses)
    
    def add_electrical_engineering_courses(self) -> bool:
        """
        Add Electrical Engineering courses to knowledge base.
        Includes 10 courses covering electronics, power systems, circuits, and renewable energy.
        
        Returns:
            True if successful, False otherwise
        """
        elec_eng_courses = [
            {
                "title": "Introduction to Electronics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electronics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electronics", "Circuits", "Components", "Basics"],
                "description": "Course introducing electronics including circuits, components, and basics"
            },
            {
                "title": "Electric Power Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electric-power-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Power Systems", "Electrical Power", "Grid", "Transmission"],
                "description": "Course covering electric power systems including grid and transmission"
            },
            {
                "title": "Power Electronics Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/power-electronics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Power Electronics", "Converters", "Inverters", "Switching"],
                "description": "Specialization covering power electronics including converters, inverters, and switching"
            },
            {
                "title": "Linear Circuits 1: DC Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/linear-circuits-dcanalysis",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Circuits", "DC Analysis", "Circuit Analysis", "Ohm's Law"],
                "description": "Course covering linear circuits DC analysis including circuit analysis and Ohm's law"
            },
            {
                "title": "Linear Circuits 2: AC Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/linear-circuits-ac-analysis",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Circuits", "AC Analysis", "AC Circuits", "Phasors"],
                "description": "Course covering linear circuits AC analysis including AC circuits and phasors"
            },
            {
                "title": "Introduction to Electricity and Magnetism",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electricity-magnetism",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electricity", "Magnetism", "Electromagnetism", "Physics"],
                "description": "Course introducing electricity and magnetism including electromagnetism and physics"
            },
            {
                "title": "Solar Energy Basics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/solar-energy-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "Photovoltaics", "Renewable Energy", "Solar Systems"],
                "description": "Course covering solar energy basics including photovoltaics and solar systems"
            },
            {
                "title": "Electrical Power Distribution",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electrical-power-distribution",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Power Distribution", "Electrical Distribution", "Grid Systems"],
                "description": "Course covering electrical power distribution including grid systems"
            },
            {
                "title": "Battery Management Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/battery-management-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Battery Management", "BMS", "Energy Storage", "Battery Systems"],
                "description": "Specialization covering battery management systems including energy storage and battery systems"
            },
            {
                "title": "Wind Energy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wind-energy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Wind Energy", "Renewable Energy", "Wind Turbines", "Wind Power"],
                "description": "Course covering wind energy including wind turbines and wind power"
            }
        ]
        
        log.info(f"📚 Adding {len(elec_eng_courses)} Electrical Engineering courses to knowledge base...")
        return self.batch_add_courses(elec_eng_courses)
    
    def add_electronics_courses(self) -> bool:
        """
        Add Electronics (VLSI, FPGA, PCB Design) courses to knowledge base.
        Includes 10 courses covering VLSI design, FPGA development, PCB design, and semiconductor physics.
        
        Returns:
            True if successful, False otherwise
        """
        electronics_courses = [
            {
                "title": "FPGA Design for Embedded Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/fpga-design",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FPGA", "Hardware Design", "Verilog", "VHDL"],
                "description": "Specialization covering FPGA design for embedded systems including Verilog and VHDL"
            },
            {
                "title": "Introduction to FPGA Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intro-fpga-design-embedded-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FPGA", "FPGA Design", "Introduction", "Basics"],
                "description": "Course introducing FPGA design including basics"
            },
            {
                "title": "Hardware Description Languages for FPGA",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/fpga-hardware-description-languages",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HDL", "Verilog", "VHDL", "FPGA Programming"],
                "description": "Course covering hardware description languages for FPGA including Verilog and VHDL"
            },
            {
                "title": "Crash Course Electronics and PCB Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/crash-course-electronics-and-pcb-design/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electronics", "PCB Design", "Circuit Design", "PCB Layout"],
                "description": "Course covering crash course in electronics and PCB design including circuit design and PCB layout"
            },
            {
                "title": "Learn PCB Design with Eagle",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/learn-pcb-design-with-eagle/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PCB Design", "Eagle", "PCB Layout", "Schematic Design"],
                "description": "Course covering PCB design with Eagle including PCB layout and schematic design"
            },
            {
                "title": "VLSI CAD Part I: Logic",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/vlsi-cad-logic",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VLSI", "CAD", "Logic Design", "Digital Design"],
                "description": "Course covering VLSI CAD Part I including logic design and digital design"
            },
            {
                "title": "VLSI CAD Part II: Layout",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/vlsi-cad-layout",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VLSI", "CAD", "Layout Design", "Physical Design"],
                "description": "Course covering VLSI CAD Part II including layout design and physical design"
            },
            {
                "title": "Mastering Altium Designer: PCB Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-altium-designer/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Altium Designer", "PCB Design", "Professional PCB", "Advanced Design"],
                "description": "Course covering mastering Altium Designer for professional PCB design"
            },
            {
                "title": "Semiconductor Physics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/semiconductor-physics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Semiconductor Physics", "Solid State Physics", "Device Physics"],
                "description": "Course covering semiconductor physics including solid state physics and device physics"
            },
            {
                "title": "Transistor - Field Effect Transistor",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/transistor-field-effect-transistor",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Transistors", "FET", "MOSFET", "Device Physics"],
                "description": "Course covering transistors including field effect transistors (FET) and MOSFET"
            }
        ]
        
        log.info(f"📚 Adding {len(electronics_courses)} Electronics courses to knowledge base...")
        return self.batch_add_courses(electronics_courses)
    
    def add_cloud_hardware_courses(self) -> bool:
        """
        Add Cloud Hardware (Server Systems & Data Centers) courses to knowledge base.
        Includes 10 courses covering computer architecture, server administration, data centers, and IT infrastructure.
        
        Returns:
            True if successful, False otherwise
        """
        cloud_hardware_courses = [
            {
                "title": "Computer Architecture",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/computer-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Architecture", "CPU Design", "Memory Systems", "Processor Design"],
                "description": "Course covering computer architecture including CPU design, memory systems, and processor design"
            },
            {
                "title": "Introduction to Hardware and Operating Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-hardware-and-operating-systems",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hardware", "Operating Systems", "Computer Hardware", "OS Basics"],
                "description": "Course introducing hardware and operating systems including computer hardware and OS basics"
            },
            {
                "title": "Google IT Support (Hardware Module)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-it-support",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["IT Support", "Hardware", "Troubleshooting", "Computer Hardware"],
                "description": "Professional certificate covering Google IT support including hardware module and troubleshooting"
            },
            {
                "title": "Windows Server 2019 Administration",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/windows-server-2019-administration-h/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Windows Server", "Server Administration", "Active Directory", "Server Management"],
                "description": "Course covering Windows Server 2019 administration including Active Directory and server management"
            },
            {
                "title": "Linux Server Management and Security",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/linux-server-management-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linux Server", "Server Management", "Security", "System Administration"],
                "description": "Course covering Linux server management and security including system administration"
            },
            {
                "title": "Data Center Essentials: General Introduction",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/data-center-essentials-general-introduction/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Centers", "Infrastructure", "Data Center Operations", "Facilities"],
                "description": "Course covering data center essentials including infrastructure and data center operations"
            },
            {
                "title": "Introduction to Server Administration",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/introduction-to-server-administration/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Server Administration", "Introduction", "Server Basics"],
                "description": "Course introducing server administration including server basics"
            },
            {
                "title": "Building a Future with Robots (Hardware)",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/robotics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Robotics Hardware", "Robot Systems", "Hardware Components"],
                "description": "Course covering building a future with robots including robotics hardware and components"
            },
            {
                "title": "Fundamentals of Computer Network Security",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/computer-network-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Network Security", "Computer Security", "Infrastructure Security"],
                "description": "Specialization covering fundamentals of computer network security including infrastructure security"
            },
            {
                "title": "System Administration and IT Infrastructure",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/system-administration-it-infrastructure-services",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["System Administration", "IT Infrastructure", "Infrastructure Services", "Server Management"],
                "description": "Course covering system administration and IT infrastructure including infrastructure services"
            }
        ]
        
        log.info(f"📚 Adding {len(cloud_hardware_courses)} Cloud Hardware courses to knowledge base...")
        return self.batch_add_courses(cloud_hardware_courses)
    
    def add_mechanical_engineering_courses_set2(self) -> bool:
        """
        Add Mechanical Engineering courses (Set 2) to knowledge base.
        Includes 10 additional courses covering CAD/CAM/CAE, CFD, Six Sigma, MATLAB, and specialized topics.
        
        Returns:
            True if successful, False otherwise
        """
        mech_eng_courses = [
            {
                "title": "Autodesk CAD/CAM/CAE for Mechanical Engineering",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/autodesk-cad-cam-cae-mechanical-engineering",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CAD", "CAM", "CAE", "Autodesk", "Mechanical Design"],
                "description": "Specialization covering Autodesk CAD/CAM/CAE for mechanical engineering including mechanical design"
            },
            {
                "title": "Computational Fluid Dynamics (CFD) Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/fluid-mechanics-and-cfd-fundamentals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CFD", "Fluid Mechanics", "Simulation", "Numerical Methods"],
                "description": "Course covering computational fluid dynamics fundamentals including fluid mechanics and simulation"
            },
            {
                "title": "Introduction to Engineering (Mechanics Focus)",
                "provider": "edX",
                "url": "https://www.edx.org/course/a-hands-on-introduction-to-engineering-simulations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering", "Mechanics", "Simulations", "Hands-on"],
                "description": "Course introducing engineering with mechanics focus including hands-on simulations"
            },
            {
                "title": "Six Sigma Green Belt Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/six-sigma-green-belt",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Six Sigma", "Quality Management", "Process Improvement", "DMAIC"],
                "description": "Specialization covering Six Sigma Green Belt including quality management and process improvement"
            },
            {
                "title": "MATLAB Programming for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/matlab-programming-engineers-scientists",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["MATLAB", "Programming", "Engineering", "Scientific Computing"],
                "description": "Specialization covering MATLAB programming for engineers including scientific computing"
            },
            {
                "title": "Fundamentals of Fluid Power",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/fluid-power",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Fluid Power", "Hydraulics", "Pneumatics", "Fluid Systems"],
                "description": "Course covering fundamentals of fluid power including hydraulics and pneumatics"
            },
            {
                "title": "SolidWorks: Become a Certified Associate (CSWA)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/solidworks-cswa/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SolidWorks", "CSWA", "3D CAD", "Certification"],
                "description": "Course covering SolidWorks to become a certified associate (CSWA) including 3D CAD"
            },
            {
                "title": "Introduction to Geometric Dimensioning (GD&T)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/geometric-dimensioning-and-tolerancing-gdt-fundamentals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GD&T", "Geometric Dimensioning", "Tolerancing", "Manufacturing"],
                "description": "Course introducing geometric dimensioning and tolerancing (GD&T) including manufacturing"
            },
            {
                "title": "Ferrous Technology I (Steel Metallurgy)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ferrous-technology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Steel Metallurgy", "Materials Science", "Ferrous Metals", "Metallurgy"],
                "description": "Course covering ferrous technology including steel metallurgy and materials science"
            },
            {
                "title": "Control of Mobile Robots",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mobile-robot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mobile Robots", "Control Systems", "Robotics", "Navigation"],
                "description": "Course covering control of mobile robots including control systems and navigation"
            }
        ]
        
        log.info(f"📚 Adding {len(mech_eng_courses)} Mechanical Engineering (Set 2) courses to knowledge base...")
        return self.batch_add_courses(mech_eng_courses)
    
    def add_electrical_engineering_courses_set2(self) -> bool:
        """
        Add Electrical Engineering courses (Set 2) to knowledge base.
        Includes 10 additional courses covering semiconductors, smart grids, electric vehicles, and advanced topics.
        
        Returns:
            True if successful, False otherwise
        """
        elec_eng_courses = [
            {
                "title": "Introduction to Electronics: Semiconductors",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electronics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electronics", "Semiconductors", "Device Physics", "Electronic Devices"],
                "description": "Course introducing electronics with focus on semiconductors including device physics"
            },
            {
                "title": "Electric Industry Operations and Markets",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electric-industry-operations-markets",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electric Industry", "Power Markets", "Operations", "Energy Markets"],
                "description": "Course covering electric industry operations and markets including power markets"
            },
            {
                "title": "Fundamentals of Electrical Controls",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/fundamentals-of-electrical-controls/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electrical Controls", "Control Systems", "Automation", "PLC"],
                "description": "Course covering fundamentals of electrical controls including control systems and automation"
            },
            {
                "title": "Smart Grid: Fundamentals and Technologies",
                "provider": "edX",
                "url": "https://www.edx.org/learn/smart-grids/delft-university-of-technology-smart-grids-fundamentals-and-technologies",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Smart Grid", "Grid Technology", "Energy Systems", "Smart Infrastructure"],
                "description": "Course covering smart grid fundamentals and technologies including smart infrastructure"
            },
            {
                "title": "Electrical Engineering: Circuit Analysis",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/electrical-engineering-circuit-analysis/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Circuit Analysis", "Electrical Engineering", "Circuit Theory", "Analysis Methods"],
                "description": "Course covering electrical engineering circuit analysis including circuit theory"
            },
            {
                "title": "Electric Vehicles and Mobility",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electric-vehicles-mobility",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electric Vehicles", "EV Technology", "Mobility", "Battery Systems"],
                "description": "Course covering electric vehicles and mobility including EV technology and battery systems"
            },
            {
                "title": "Power System Protection",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/power-system-protection-fundamentals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Power System Protection", "Protection Systems", "Relays", "Fault Analysis"],
                "description": "Course covering power system protection including protection systems and fault analysis"
            },
            {
                "title": "Introduction to Satellite Communications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/satellite-communications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Satellite Communications", "Telecommunications", "Satellite Systems"],
                "description": "Course introducing satellite communications including telecommunications and satellite systems"
            },
            {
                "title": "Electricity & Magnetism: Fields & Forces",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electricity-magnetism-fields-forces",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electricity", "Magnetism", "Fields", "Forces", "Electromagnetism"],
                "description": "Course covering electricity and magnetism including fields, forces, and electromagnetism"
            },
            {
                "title": "Solar Energy System Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/solar-energy-system-design/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "System Design", "Photovoltaics", "Solar Systems"],
                "description": "Course covering solar energy system design including photovoltaics and solar systems"
            }
        ]
        
        log.info(f"📚 Adding {len(elec_eng_courses)} Electrical Engineering (Set 2) courses to knowledge base...")
        return self.batch_add_courses(elec_eng_courses)
    
    def add_electronics_courses_set2(self) -> bool:
        """
        Add Electronics courses (Set 2) to knowledge base.
        Includes 10 additional courses covering nanoelectronics, digital systems, embedded ARM, VHDL, and sensors.
        
        Returns:
            True if successful, False otherwise
        """
        electronics_courses = [
            {
                "title": "Digital Systems: From Logic Gates to Processors",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/digital-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Systems", "Logic Gates", "Processors", "Digital Design"],
                "description": "Course covering digital systems from logic gates to processors including digital design"
            },
            {
                "title": "Embedded Systems Design with ARM",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/embedded-systems-design-with-arm/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Embedded Systems", "ARM", "ARM Cortex", "Embedded Design"],
                "description": "Course covering embedded systems design with ARM including ARM Cortex"
            },
            {
                "title": "From NAND to Tetris Part I",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/build-a-computer",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Architecture", "Hardware", "Building Computers", "Logic Design"],
                "description": "Course covering building a computer from NAND gates including hardware and logic design"
            },
            {
                "title": "VHDL for FPGA Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/vhdl-programming-for-fpga-development/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VHDL", "FPGA", "HDL Programming", "FPGA Development"],
                "description": "Course covering VHDL for FPGA development including HDL programming"
            },
            {
                "title": "SystemVerilog for Verification",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/systemverilog-for-verification-part-1-fundamentals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SystemVerilog", "Verification", "HDL", "Testbench"],
                "description": "Course covering SystemVerilog for verification including testbench development"
            },
            {
                "title": "CMOS Digital VLSI Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cmos-digital-vlsi-design/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CMOS", "VLSI Design", "Digital Design", "IC Design"],
                "description": "Course covering CMOS digital VLSI design including IC design"
            },
            {
                "title": "Introduction to Digital Signal Processing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/digital-signal-processing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Signal Processing", "DSP", "Signal Processing", "Filters"],
                "description": "Course introducing digital signal processing including filters and signal processing"
            },
            {
                "title": "Sensors and Sensor Circuit Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/sensors-circuit-interface",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sensors", "Sensor Circuits", "Interface Design", "Sensor Systems"],
                "description": "Course covering sensors and sensor circuit design including interface design"
            },
            {
                "title": "The Arduino Platform and C Programming",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/arduino-platform",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Arduino", "C Programming", "Embedded Programming", "Microcontrollers"],
                "description": "Course covering Arduino platform and C programming including embedded programming"
            }
        ]
        
        log.info(f"📚 Adding {len(electronics_courses)} Electronics (Set 2) courses to knowledge base...")
        return self.batch_add_courses(electronics_courses)
    
    def add_cloud_hardware_courses_set2(self) -> bool:
        """
        Add Cloud Hardware & Server Systems courses (Set 2) to knowledge base.
        Includes 10 additional courses covering server administration, virtualization, data centers, and hardware security.
        
        Returns:
            True if successful, False otherwise
        """
        cloud_hardware_courses = [
            {
                "title": "CompTIA Server+ Certification Training",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/comptia-server-plus-training/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CompTIA Server+", "Server Certification", "Server Administration"],
                "description": "Training course for CompTIA Server+ certification including server administration"
            },
            {
                "title": "Data Center Cooling Essentials",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/data-center-cooling-essentials/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Center", "Cooling Systems", "HVAC", "Data Center Operations"],
                "description": "Course covering data center cooling essentials including HVAC and data center operations"
            },
            {
                "title": "Virtualization with VMware vSphere",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/virtualization-with-vmware-vsphere-6-7-bootcamp/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["VMware", "vSphere", "Virtualization", "Server Virtualization"],
                "description": "Course covering virtualization with VMware vSphere including server virtualization"
            },
            {
                "title": "Fundamentals of Red Hat Enterprise Linux",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/fundamentals-red-hat-enterprise-linux",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Red Hat", "RHEL", "Linux", "Enterprise Linux"],
                "description": "Course covering fundamentals of Red Hat Enterprise Linux including enterprise Linux administration"
            },
            {
                "title": "Operating Systems and You: Power User",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/os-power-user",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operating Systems", "Power User", "OS Administration", "System Management"],
                "description": "Course covering operating systems for power users including OS administration"
            },
            {
                "title": "Hardware Security",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/hardware-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Hardware Security", "Security", "Hardware Attacks", "Secure Hardware"],
                "description": "Course covering hardware security including hardware attacks and secure hardware design"
            },
            {
                "title": "Cloud Computing Infrastructure",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cloud-computing-infrastructure",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Infrastructure", "Cloud Computing", "Infrastructure", "Cloud Systems"],
                "description": "Course covering cloud computing infrastructure including cloud systems"
            },
            {
                "title": "High Performance Computing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/high-performance-computing",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HPC", "High Performance Computing", "Parallel Computing", "Supercomputing"],
                "description": "Course covering high performance computing including parallel computing and supercomputing"
            },
            {
                "title": "Server Administration Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/server-administration-fundamentals/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Server Administration", "Fundamentals", "Server Management", "Basics"],
                "description": "Course covering server administration fundamentals including server management basics"
            },
            {
                "title": "Building Your Own Computer",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/building-a-computer",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Building", "Hardware Assembly", "PC Building", "Computer Components"],
                "description": "Course covering building your own computer including hardware assembly and PC building"
            }
        ]
        
        log.info(f"📚 Adding {len(cloud_hardware_courses)} Cloud Hardware (Set 2) courses to knowledge base...")
        return self.batch_add_courses(cloud_hardware_courses)
    
    def add_communication_skills_courses(self) -> bool:
        """
        Add Communication Skills courses to knowledge base.
        Includes 10 courses covering teamwork, business writing, virtual communication, and presentation skills.
        
        Returns:
            True if successful, False otherwise
        """
        communication_courses = [
            {
                "title": "Teamwork & Communication",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teamwork-skills-effective-communication",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Teamwork", "Communication", "Collaboration", "Interpersonal Skills"],
                "description": "Course covering teamwork and effective communication skills including collaboration"
            },
            {
                "title": "Business Writing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/business-writing",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Writing", "Professional Writing", "Written Communication"],
                "description": "Course covering business writing including professional writing and written communication"
            },
            {
                "title": "Communication Strategies for a Virtual Age",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/communication-strategies-virtual-age",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Virtual Communication", "Remote Work", "Digital Communication", "Online Collaboration"],
                "description": "Course covering communication strategies for virtual age including remote work and digital communication"
            },
            {
                "title": "Improving Communication Skills",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-communication-skills",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Communication Skills", "Interpersonal Communication", "Professional Communication"],
                "description": "Course covering improving communication skills including interpersonal and professional communication"
            },
            {
                "title": "Effective Communication: Writing, Design, and Presentation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/effective-business-communication",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Communication", "Writing", "Design", "Presentation Skills"],
                "description": "Specialization covering effective communication including writing, design, and presentation skills"
            },
            {
                "title": "Communication Skills for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/communication-skills-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Communication Skills", "Engineering Communication", "Technical Communication"],
                "description": "Specialization covering communication skills for engineers including technical communication"
            },
            {
                "title": "English for Career Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/careerdevelopment",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["English", "Career Development", "Professional English", "Business English"],
                "description": "Course covering English for career development including professional and business English"
            },
            {
                "title": "Storytelling for Business",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/storytelling-for-business/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Storytelling", "Business Storytelling", "Narrative", "Communication"],
                "description": "Course covering storytelling for business including narrative and communication"
            },
            {
                "title": "Communication Foundations",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/communication-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Communication", "Foundations", "Basic Communication", "Professional Communication"],
                "description": "Course covering communication foundations including basic and professional communication"
            },
            {
                "title": "Workplace Communication: You Can Speak Up",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/workplace-communication-you-can-speak-up",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Workplace Communication", "Speaking Up", "Professional Communication", "Assertiveness"],
                "description": "Course covering workplace communication including speaking up and assertiveness"
            }
        ]
        
        log.info(f"📚 Adding {len(communication_courses)} Communication Skills courses to knowledge base...")
        return self.batch_add_courses(communication_courses)
    
    def add_leadership_courses(self) -> bool:
        """
        Add Leadership courses to knowledge base.
        Includes 10 courses covering strategic leadership, team management, agile leadership, and organizational leadership.
        
        Returns:
            True if successful, False otherwise
        """
        leadership_courses = [
            {
                "title": "Strategic Leadership and Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/strategic-leadership",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategic Leadership", "Management", "Leadership Strategy", "Business Leadership"],
                "description": "Specialization covering strategic leadership and management including leadership strategy"
            },
            {
                "title": "Leading People and Teams",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/leading-teams",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Team Leadership", "People Management", "Team Management"],
                "description": "Specialization covering leading people and teams including team and people management"
            },
            {
                "title": "Exercising Leadership: Foundational Principles",
                "provider": "edX",
                "url": "https://www.edx.org/course/exercising-leadership-foundational-principles",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Leadership Principles", "Foundational Leadership"],
                "description": "Course covering exercising leadership including foundational principles"
            },
            {
                "title": "Foundations of Everyday Leadership",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/everyday-leadership-foundation",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Everyday Leadership", "Leadership Foundations"],
                "description": "Course covering foundations of everyday leadership"
            },
            {
                "title": "Leadership: Practical Skills",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-leadership-skills",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Practical Leadership", "Leadership Skills"],
                "description": "Course covering leadership practical skills"
            },
            {
                "title": "Agile Leadership Principles",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/agile-leadership",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Agile Leadership", "Agile", "Leadership Principles"],
                "description": "Course covering agile leadership principles"
            },
            {
                "title": "Organizational Leadership",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/organizational-leadership",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Organizational Leadership", "Leadership", "Organizational Management"],
                "description": "Specialization covering organizational leadership"
            },
            {
                "title": "Google Project Management (Leadership focus)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/google-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Leadership", "Google PM", "Management"],
                "description": "Professional certificate covering Google project management with leadership focus"
            },
            {
                "title": "Becoming a Successful Leader",
                "provider": "edX",
                "url": "https://www.edx.org/course/becoming-a-successful-leader",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Success", "Leadership Development"],
                "description": "Course covering becoming a successful leader including leadership development"
            },
            {
                "title": "Think Like a Leader",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/think-like-a-leader/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Leadership", "Leadership Mindset", "Strategic Thinking"],
                "description": "Course covering thinking like a leader including leadership mindset and strategic thinking"
            }
        ]
        
        log.info(f"📚 Adding {len(leadership_courses)} Leadership courses to knowledge base...")
        return self.batch_add_courses(leadership_courses)
    
    def add_resume_interview_courses(self) -> bool:
        """
        Add Resume & Interview Prep courses to knowledge base.
        Includes 10 courses covering resume writing, cover letters, technical interviews, and career success.
        
        Returns:
            True if successful, False otherwise
        """
        resume_interview_courses = [
            {
                "title": "How to Write a Resume",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/how-to-write-a-resume",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Resume Writing", "Resume", "Job Application", "Career"],
                "description": "Course covering how to write a resume including job application and career"
            },
            {
                "title": "Writing Winning Resumes and Cover Letters",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/writing-winning-resumes-and-cover-letters",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Resume Writing", "Cover Letters", "Job Application", "Career"],
                "description": "Course covering writing winning resumes and cover letters"
            },
            {
                "title": "Interviewing for a Technical Job",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/technical-interview",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technical Interview", "Interviewing", "Job Interview", "Tech Interview"],
                "description": "Course covering interviewing for a technical job including tech interview preparation"
            },
            {
                "title": "Mastering the Software Engineering Interview",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cs-tech-interview",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Software Engineering Interview", "Tech Interview", "Coding Interview", "Interview Prep"],
                "description": "Course covering mastering the software engineering interview including coding interview prep"
            },
            {
                "title": "Tech Interview Pro",
                "provider": "TechLead",
                "url": "https://www.techinterviewpro.com/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tech Interview", "Interview Prep", "Technical Interview", "Coding Interview"],
                "description": "Course covering tech interview pro including technical and coding interview preparation"
            },
            {
                "title": "Career Success Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/career-success",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Career Success", "Career Development", "Professional Development"],
                "description": "Specialization covering career success including career and professional development"
            },
            {
                "title": "Rock the Tech Interview",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/rock-the-tech-interview/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tech Interview", "Interview Prep", "Technical Interview"],
                "description": "Course covering rock the tech interview including interview preparation"
            },
            {
                "title": "Successful Interviewing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/successful-interviewing",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Interviewing", "Interview Skills", "Job Interview"],
                "description": "Course covering successful interviewing including interview skills"
            },
            {
                "title": "Advanced Interviewing Techniques",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/advanced-interviewing-techniques",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Interviewing", "Advanced Interviewing", "Interview Techniques"],
                "description": "Course covering advanced interviewing techniques"
            },
            {
                "title": "Complete Resume, LinkedIn & Interview Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/complete-resume-linkedin-interview-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Resume", "LinkedIn", "Interview Prep", "Career"],
                "description": "Course covering complete resume, LinkedIn and interview guide"
            }
        ]
        
        log.info(f"📚 Adding {len(resume_interview_courses)} Resume & Interview Prep courses to knowledge base...")
        return self.batch_add_courses(resume_interview_courses)
    
    def add_public_speaking_courses(self) -> bool:
        """
        Add Public Speaking courses to knowledge base.
        Includes 10 courses covering dynamic public speaking, presentation skills, rhetoric, and storytelling.
        
        Returns:
            True if successful, False otherwise
        """
        public_speaking_courses = [
            {
                "title": "Dynamic Public Speaking",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/public-speaking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Presentation", "Speaking Skills"],
                "description": "Specialization covering dynamic public speaking including presentation and speaking skills"
            },
            {
                "title": "Introduction to Public Speaking",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/public-speaking",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Introduction", "Speaking Basics"],
                "description": "Course introducing public speaking including speaking basics"
            },
            {
                "title": "Presentation Skills: Speechwriting and Delivery",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/presentation-skills",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Presentation Skills", "Speechwriting", "Delivery", "Public Speaking"],
                "description": "Specialization covering presentation skills including speechwriting and delivery"
            },
            {
                "title": "Public Speaking for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/public-speaking-for-beginners-al/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Beginners", "Speaking Basics"],
                "description": "Course covering public speaking for beginners"
            },
            {
                "title": "Rhetoric: The Art of Persuasive Writing and Public Speaking",
                "provider": "edX",
                "url": "https://www.edx.org/course/rhetoric-the-art-of-persuasive-writing-and-public-speaking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Rhetoric", "Persuasive Writing", "Public Speaking", "Persuasion"],
                "description": "Course covering rhetoric including persuasive writing and public speaking"
            },
            {
                "title": "Complete Public Speaking Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-public-speaking-course-2018/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Masterclass", "Speaking Skills"],
                "description": "Course covering complete public speaking masterclass"
            },
            {
                "title": "Speaking to inform: Discussing complex ideas",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/speak-to-inform",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Informative Speaking", "Complex Ideas"],
                "description": "Course covering speaking to inform including discussing complex ideas"
            },
            {
                "title": "Presentations: Speaking so that People Listen",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/presentations-speaking-so-that-people-listen",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Presentations", "Public Speaking", "Communication"],
                "description": "Course covering presentations including speaking so that people listen"
            },
            {
                "title": "Storytelling and influencing: Communicate with impact",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/storytelling-influence",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Storytelling", "Influencing", "Communication", "Impact"],
                "description": "Course covering storytelling and influencing including communicating with impact"
            },
            {
                "title": "Public Speaking: You Can Speak to Anyone",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/public-speaking-you-can-speak-to-anyone/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Speaking", "Speaking Skills", "Confidence"],
                "description": "Course covering public speaking including speaking to anyone with confidence"
            }
        ]
        
        log.info(f"📚 Adding {len(public_speaking_courses)} Public Speaking courses to knowledge base...")
        return self.batch_add_courses(public_speaking_courses)
    
    def add_negotiation_courses(self) -> bool:
        """
        Add Negotiation courses to knowledge base.
        Includes 10 courses covering negotiation strategies, salary negotiation, business negotiation, and international negotiation.
        
        Returns:
            True if successful, False otherwise
        """
        negotiation_courses = [
            {
                "title": "Successful Negotiation: Essential Strategies and Skills",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/negotiation-skills",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Negotiation Strategies", "Negotiation Skills"],
                "description": "Course covering successful negotiation including essential strategies and skills"
            },
            {
                "title": "Introduction to Negotiation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/negotiation",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Introduction", "Negotiation Basics"],
                "description": "Course introducing negotiation including negotiation basics"
            },
            {
                "title": "Negotiation Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/negotiation-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Fundamentals", "Negotiation Basics"],
                "description": "Course covering negotiation fundamentals"
            },
            {
                "title": "Art of Negotiation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/art-of-negotiation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Art of Negotiation", "Negotiation Techniques"],
                "description": "Course covering art of negotiation including negotiation techniques"
            },
            {
                "title": "Negotiation Mastery",
                "provider": "Harvard Online",
                "url": "https://online.hbs.edu/courses/negotiation-mastery/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Negotiation Mastery", "Advanced Negotiation"],
                "description": "Course covering negotiation mastery including advanced negotiation"
            },
            {
                "title": "Negotiation Skills: Negotiate Your Salary",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/negotiation-skills-negotiate-your-salary/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Salary Negotiation", "Career"],
                "description": "Course covering negotiation skills including negotiating your salary"
            },
            {
                "title": "Negotiating for Business Success",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/negotiating-for-business-success",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Business Negotiation", "Business Success"],
                "description": "Course covering negotiating for business success"
            },
            {
                "title": "Mastering Negotiations",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-negotiations/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Mastering Negotiations", "Negotiation Skills"],
                "description": "Course covering mastering negotiations"
            },
            {
                "title": "High Performance Collaboration: Leadership, Teamwork, and Negotiation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/leadership-collaboration",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Collaboration", "Leadership", "Teamwork", "Negotiation"],
                "description": "Course covering high performance collaboration including leadership, teamwork, and negotiation"
            },
            {
                "title": "International Business and Culture",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/international-business-culture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["International Business", "Culture", "Cross-Cultural", "Global Business"],
                "description": "Course covering international business and culture including cross-cultural and global business"
            }
        ]
        
        log.info(f"📚 Adding {len(negotiation_courses)} Negotiation courses to knowledge base...")
        return self.batch_add_courses(negotiation_courses)
    
    def add_critical_thinking_courses(self) -> bool:
        """
        Add Critical Thinking courses to knowledge base.
        Includes 10 courses covering logic, problem solving, decision making, and creative thinking.
        
        Returns:
            True if successful, False otherwise
        """
        critical_thinking_courses = [
            {
                "title": "Introduction to Logic and Critical Thinking",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/logical-critical-thinking",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Logic", "Critical Thinking", "Logical Thinking", "Reasoning"],
                "description": "Course introducing logic and critical thinking including logical thinking and reasoning"
            },
            {
                "title": "Critical Thinking & Problem Solving",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/critical-thinking-problem-solving",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Critical Thinking", "Problem Solving", "Analytical Thinking"],
                "description": "Course covering critical thinking and problem solving including analytical thinking"
            },
            {
                "title": "Problem Solving and Critical Thinking Skills",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/problem-solving-critical-thinking-skills/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Problem Solving", "Critical Thinking", "Thinking Skills"],
                "description": "Course covering problem solving and critical thinking skills"
            },
            {
                "title": "Mindware: Critical Thinking for the Information Age",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mindware",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Critical Thinking", "Information Age", "Mindware", "Analytical Thinking"],
                "description": "Course covering mindware including critical thinking for the information age"
            },
            {
                "title": "Creative Thinking: Techniques and Tools for Success",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/creative-thinking-techniques-and-tools-for-success",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Creative Thinking", "Innovation", "Problem Solving", "Creativity"],
                "description": "Course covering creative thinking including techniques and tools for success"
            },
            {
                "title": "Critical Thinking for the Information Age",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/critical-thinking-information-age",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Critical Thinking", "Information Age", "Analytical Thinking"],
                "description": "Course covering critical thinking for the information age"
            },
            {
                "title": "Philosophy and Critical Thinking",
                "provider": "edX",
                "url": "https://www.edx.org/course/philosophy-and-critical-thinking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Philosophy", "Critical Thinking", "Logical Reasoning"],
                "description": "Course covering philosophy and critical thinking including logical reasoning"
            },
            {
                "title": "Making Better Decisions",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/making-better-decisions",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Decision Making", "Better Decisions", "Problem Solving"],
                "description": "Course covering making better decisions including problem solving"
            },
            {
                "title": "Effective Problem-Solving and Decision-Making",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/problem-solving-decision-making",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Problem Solving", "Decision Making", "Effective Problem Solving"],
                "description": "Course covering effective problem-solving and decision-making"
            },
            {
                "title": "Mastering Critical Thinking",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-critical-thinking/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Critical Thinking", "Mastering Critical Thinking", "Analytical Skills"],
                "description": "Course covering mastering critical thinking including analytical skills"
            }
        ]
        
        log.info(f"📚 Adding {len(critical_thinking_courses)} Critical Thinking courses to knowledge base...")
        return self.batch_add_courses(critical_thinking_courses)
    
    def add_b2b_sales_courses(self) -> bool:
        """
        Add B2B Sales Strategies courses to knowledge base.
        Includes 10 courses covering sales techniques, negotiation, sales management, and B2B sales.
        
        Returns:
            True if successful, False otherwise
        """
        b2b_sales_courses = [
            {
                "title": "The Art of Sales: Mastering the Selling Process",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/art-of-sales",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales", "Selling Process", "Sales Techniques", "B2B Sales"],
                "description": "Specialization covering the art of sales including mastering the selling process"
            },
            {
                "title": "Sales Training: Practical Sales Techniques",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sales-training-practical-sales-techniques/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales Training", "Sales Techniques", "Practical Sales"],
                "description": "Course covering sales training including practical sales techniques"
            },
            {
                "title": "Inbound Sales Certification",
                "provider": "HubSpot Academy",
                "url": "https://academy.hubspot.com/courses/inbound-sales",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Inbound Sales", "Sales Certification", "HubSpot", "Sales"],
                "description": "Course covering inbound sales certification including HubSpot sales"
            },
            {
                "title": "B2B Sales Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/b2b-sales-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["B2B Sales", "Sales Masterclass", "Business Sales"],
                "description": "Course covering B2B sales masterclass including business sales"
            },
            {
                "title": "Enterprise Sales Certificate",
                "provider": "Queen's University",
                "url": "https://smith.queensu.ca/execed/programs/enterprise-sales-certificate.php",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Enterprise Sales", "Sales Certificate", "B2B Sales"],
                "description": "Course covering enterprise sales certificate including B2B sales"
            },
            {
                "title": "Strategic Sales Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/strategic-sales-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales Management", "Strategic Sales", "Sales Strategy"],
                "description": "Specialization covering strategic sales management including sales strategy"
            },
            {
                "title": "Successful Negotiation: Essential Strategies",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/negotiation-skills",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Negotiation", "Negotiation Strategies", "Sales Negotiation"],
                "description": "Course covering successful negotiation including essential strategies"
            },
            {
                "title": "Sales Machine: The B2B Sales Training",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sales-machine-b2b-sales-training/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["B2B Sales", "Sales Training", "Sales Machine"],
                "description": "Course covering sales machine including B2B sales training"
            },
            {
                "title": "Asking Great Sales Questions",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/asking-great-sales-questions",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales Questions", "Sales Skills", "Questioning Techniques"],
                "description": "Course covering asking great sales questions including questioning techniques"
            },
            {
                "title": "Introduction to Sales",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-sales",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales", "Introduction to Sales", "Sales Basics"],
                "description": "Course introducing sales including sales basics"
            }
        ]
        
        log.info(f"📚 Adding {len(b2b_sales_courses)} B2B Sales Strategies courses to knowledge base...")
        return self.batch_add_courses(b2b_sales_courses)
    
    def add_crm_courses(self) -> bool:
        """
        Add Customer Relationship Management (CRM) courses to knowledge base.
        Includes 10 courses covering Salesforce, HubSpot, Microsoft Dynamics, Zoho, and CRM fundamentals.
        
        Returns:
            True if successful, False otherwise
        """
        crm_courses = [
            {
                "title": "Salesforce Sales Operations Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/salesforce-sales-operations",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Salesforce", "Sales Operations", "CRM", "Salesforce Certification"],
                "description": "Professional certificate covering Salesforce sales operations including CRM"
            },
            {
                "title": "The Complete Salesforce Administrator Certification",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/salesforce-administrator/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Salesforce", "Salesforce Administrator", "CRM", "Salesforce Certification"],
                "description": "Course covering complete Salesforce administrator certification including CRM"
            },
            {
                "title": "HubSpot Sales Software Certification",
                "provider": "HubSpot Academy",
                "url": "https://academy.hubspot.com/courses/sales-software",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HubSpot", "Sales Software", "CRM", "HubSpot Certification"],
                "description": "Course covering HubSpot sales software certification including CRM"
            },
            {
                "title": "Salesforce Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/salesforce-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Salesforce", "Salesforce Fundamentals", "CRM"],
                "description": "Specialization covering Salesforce fundamentals including CRM"
            },
            {
                "title": "Microsoft Dynamics 365 Fundamentals (CRM)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/microsoft-dynamics-365-fundamentals-crm-mb-910/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Microsoft Dynamics 365", "CRM", "Dynamics CRM", "Microsoft CRM"],
                "description": "Course covering Microsoft Dynamics 365 fundamentals including CRM"
            },
            {
                "title": "HubSpot CRM Essentials",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/hubspot-crm-essentials",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HubSpot", "CRM", "HubSpot CRM", "CRM Essentials"],
                "description": "Course covering HubSpot CRM essentials"
            },
            {
                "title": "Salesforce Administrator Professional Certificate",
                "provider": "Coursera",
                "url": "https://www.coursera.org/professional-certificates/salesforce-administrator",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Salesforce", "Salesforce Administrator", "CRM", "Professional Certificate"],
                "description": "Professional certificate covering Salesforce administrator including CRM"
            },
            {
                "title": "CRM Fundamentals",
                "provider": "edX",
                "url": "https://www.edx.org/learn/customer-relationship-management/babson-college-customer-relationship-management",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CRM", "Customer Relationship Management", "CRM Fundamentals"],
                "description": "Course covering CRM fundamentals including customer relationship management"
            },
            {
                "title": "Zoho CRM Administrator",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/zoho-crm-administrator/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Zoho CRM", "CRM", "Zoho", "CRM Administration"],
                "description": "Course covering Zoho CRM administrator including CRM administration"
            },
            {
                "title": "Managing the Sales Process with HubSpot",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/managing-sales-hubspot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HubSpot", "Sales Process", "CRM", "Sales Management"],
                "description": "Course covering managing the sales process with HubSpot including CRM"
            }
        ]
        
        log.info(f"📚 Adding {len(crm_courses)} Customer Relationship Management (CRM) courses to knowledge base...")
        return self.batch_add_courses(crm_courses)
    
    def add_customer_success_courses(self) -> bool:
        """
        Add Customer Success Manager (CSM) courses to knowledge base.
        Includes 10 courses covering customer success, engagement, retention, customer service, and customer experience.
        
        Returns:
            True if successful, False otherwise
        """
        customer_success_courses = [
            {
                "title": "Customer Success Management Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/customer-success-manager/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Success", "Customer Success Manager", "CSM", "Customer Management"],
                "description": "Course covering customer success management fundamentals including CSM"
            },
            {
                "title": "Certified Customer Success Manager (CCSM)",
                "provider": "SuccessHACKER",
                "url": "https://www.successhacker.co/certified-customer-success-manager-level-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Success", "CCSM", "Customer Success Manager", "Certification"],
                "description": "Course covering certified customer success manager including CCSM certification"
            },
            {
                "title": "Customer Success Manager: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/customer-success-manager-complete-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Success", "Customer Success Manager", "CSM", "Complete Guide"],
                "description": "Course covering customer success manager including the complete guide"
            },
            {
                "title": "Engagement & Retention (Customer Success)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-engagement-retention",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Engagement", "Customer Retention", "Customer Success"],
                "description": "Course covering engagement and retention including customer success"
            },
            {
                "title": "Customer Service Professional",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/customer-service-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Service", "Customer Service Professional", "Service Skills"],
                "description": "Course covering customer service professional including service skills"
            },
            {
                "title": "Gainsight Customer Success Administrator",
                "provider": "Gainsight University",
                "url": "https://university.gainsight.com/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Gainsight", "Customer Success", "CSM", "Customer Success Platform"],
                "description": "Course covering Gainsight customer success administrator including customer success platform"
            },
            {
                "title": "Managing Customer Relationships",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-managing-customer-relationships",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Relationships", "Customer Management", "Relationship Management"],
                "description": "Course covering managing customer relationships including relationship management"
            },
            {
                "title": "Customer Experience: Journey Mapping",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/customer-experience-journey-mapping",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Customer Experience", "Journey Mapping", "CX", "Customer Journey"],
                "description": "Course covering customer experience including journey mapping and customer journey"
            },
            {
                "title": "Cisco Customer Success Manager (DTCSM)",
                "provider": "Cisco",
                "url": "https://www.cisco.com/c/en/us/training-events/training-certifications/certifications/specialist/customer-success/custom-success-manager.html",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cisco", "Customer Success", "DTCSM", "Customer Success Manager"],
                "description": "Course covering Cisco customer success manager including DTCSM certification"
            },
            {
                "title": "Tech Support & Customer Service",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/technical-support-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tech Support", "Customer Service", "Technical Support", "Support Skills"],
                "description": "Course covering tech support and customer service including technical support"
            }
        ]
        
        log.info(f"📚 Adding {len(customer_success_courses)} Customer Success Manager (CSM) courses to knowledge base...")
        return self.batch_add_courses(customer_success_courses)
    
    def add_sales_engineering_courses(self) -> bool:
        """
        Add Sales Engineering (Technical Sales) courses to knowledge base.
        Includes 10 courses covering technical sales, pre-sales engineering, SaaS sales, and demo skills.
        
        Returns:
            True if successful, False otherwise
        """
        sales_engineering_courses = [
            {
                "title": "Sales Engineering / Pre-Sales Engineering",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sales-engineering-presales-engineering/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales Engineering", "Pre-Sales Engineering", "Technical Sales"],
                "description": "Course covering sales engineering and pre-sales engineering including technical sales"
            },
            {
                "title": "Mastering Technical Sales: The Sales Engineer",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-technical-sales-the-sales-engineer/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technical Sales", "Sales Engineering", "Sales Engineer"],
                "description": "Course covering mastering technical sales including the sales engineer"
            },
            {
                "title": "Technical Sales: The Role of the Sales Engineer",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/technical-sales-the-role-of-the-sales-engineer",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technical Sales", "Sales Engineer", "Sales Engineering"],
                "description": "Course covering technical sales including the role of the sales engineer"
            },
            {
                "title": "PreSales Solution Consulting",
                "provider": "PreSales Academy",
                "url": "https://www.presalesacademy.com/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["PreSales", "Solution Consulting", "Sales Engineering", "Technical Sales"],
                "description": "Course covering presales solution consulting including sales engineering"
            },
            {
                "title": "Engineering Project Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/engineering-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Project Management", "Project Management", "Engineering"],
                "description": "Specialization covering engineering project management"
            },
            {
                "title": "SaaS Sales Methodology",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/saas-sales/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SaaS Sales", "Sales Methodology", "Software Sales", "B2B Sales"],
                "description": "Course covering SaaS sales methodology including software sales"
            },
            {
                "title": "AWS Partner: Sales Accreditation (Business)",
                "provider": "AWS Training",
                "url": "https://aws.amazon.com/partners/training/accreditation/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AWS", "Sales Accreditation", "Cloud Sales", "AWS Partner"],
                "description": "Course covering AWS partner sales accreditation including cloud sales"
            },
            {
                "title": "Cisco Sales Essentials",
                "provider": "Cisco",
                "url": "https://www.cisco.com/c/en/us/training-events/training-certifications.html",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cisco", "Sales Essentials", "Technical Sales", "Cisco Sales"],
                "description": "Course covering Cisco sales essentials including technical sales"
            },
            {
                "title": "Technical Sales Foundations",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/technical-sales-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technical Sales", "Sales Foundations", "Sales Basics"],
                "description": "Course covering technical sales foundations including sales basics"
            },
            {
                "title": "Demo to Win! (Demo Skills)",
                "provider": "2Win! Global",
                "url": "https://2winglobal.com/demo-to-win/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Demo Skills", "Product Demo", "Sales Demo", "Presentation Skills"],
                "description": "Course covering demo to win including demo skills and product demo"
            }
        ]
        
        log.info(f"📚 Adding {len(sales_engineering_courses)} Sales Engineering (Technical Sales) courses to knowledge base...")
        return self.batch_add_courses(sales_engineering_courses)
    
    def add_digital_sales_courses(self) -> bool:
        """
        Add Digital Sales & Social Selling courses to knowledge base.
        Includes 10 courses covering social selling, LinkedIn marketing, digital sales, and virtual selling.
        
        Returns:
            True if successful, False otherwise
        """
        digital_sales_courses = [
            {
                "title": "Social Selling with LinkedIn",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/social-selling-with-linkedin",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Social Selling", "LinkedIn", "Digital Sales", "Social Media Sales"],
                "description": "Course covering social selling with LinkedIn including digital sales"
            },
            {
                "title": "Digital Sales & Marketing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/digital-marketing-sales",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Sales", "Digital Marketing", "Online Sales", "E-commerce"],
                "description": "Course covering digital sales and marketing including online sales"
            },
            {
                "title": "LinkedIn Marketing & Lead Generation",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/linkedin-marketing-lead-generation/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["LinkedIn Marketing", "Lead Generation", "Social Selling", "B2B Marketing"],
                "description": "Course covering LinkedIn marketing and lead generation including social selling"
            },
            {
                "title": "HubSpot Social Media Marketing Certification",
                "provider": "HubSpot Academy",
                "url": "https://academy.hubspot.com/courses/social-media",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["HubSpot", "Social Media Marketing", "Digital Marketing", "Social Selling"],
                "description": "Course covering HubSpot social media marketing certification including social selling"
            },
            {
                "title": "Account-Based Marketing (ABM)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-account-based-marketing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Account-Based Marketing", "ABM", "B2B Marketing", "Targeted Marketing"],
                "description": "Course covering account-based marketing including ABM and B2B marketing"
            },
            {
                "title": "Digital Body Language for Sales",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/digital-body-language-sales/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Body Language", "Sales Communication", "Virtual Sales", "Digital Sales"],
                "description": "Course covering digital body language for sales including virtual sales"
            },
            {
                "title": "Mastering LinkedIn for Lead Generation",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-linkedin-for-lead-generation/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["LinkedIn", "Lead Generation", "Social Selling", "B2B Sales"],
                "description": "Course covering mastering LinkedIn for lead generation including social selling"
            },
            {
                "title": "Sales Prospecting",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/sales-prospecting",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sales Prospecting", "Lead Generation", "Sales", "Prospecting"],
                "description": "Course covering sales prospecting including lead generation"
            },
            {
                "title": "Virtual Selling for Sales Professionals",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/virtual-selling-for-sales-professionals",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Virtual Selling", "Remote Sales", "Digital Sales", "Online Sales"],
                "description": "Course covering virtual selling for sales professionals including remote sales"
            },
            {
                "title": "Email Marketing for Sales",
                "provider": "HubSpot Academy",
                "url": "https://academy.hubspot.com/courses/email-marketing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Email Marketing", "Sales", "Email Sales", "Digital Marketing"],
                "description": "Course covering email marketing for sales including digital marketing"
            }
        ]
        
        log.info(f"📚 Adding {len(digital_sales_courses)} Digital Sales & Social Selling courses to knowledge base...")
        return self.batch_add_courses(digital_sales_courses)
    
    def add_business_law_courses(self) -> bool:
        """
        Add Business Law courses to knowledge base.
        Includes 10 courses covering contracts, employment law, corporate law, and international business law.
        
        Returns:
            True if successful, False otherwise
        """
        business_law_courses = [
            {
                "title": "Business Law for Entrepreneurs and Managers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-law-for-entrepreneurs-and-managers/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Law", "Entrepreneurship", "Legal", "Corporate Law"],
                "description": "Course covering business law for entrepreneurs and managers including corporate law"
            },
            {
                "title": "Corporate & Commercial Law I: Contracts & Employment",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/corporate-commercial-law-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Law", "Commercial Law", "Contracts", "Employment Law"],
                "description": "Course covering corporate and commercial law including contracts and employment law"
            },
            {
                "title": "American Contract Law I",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/american-contract-law-i",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Contract Law", "American Law", "Legal", "Contracts"],
                "description": "Course covering American contract law including contracts"
            },
            {
                "title": "European Business Law: Understanding the Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/european-business-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["European Business Law", "Business Law", "Legal", "International Law"],
                "description": "Course covering European business law including understanding the fundamentals"
            },
            {
                "title": "Business Law: A Community of Practice",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/business-law-community-of-practice",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Law", "Legal Practice", "Legal"],
                "description": "Course covering business law including a community of practice"
            },
            {
                "title": "Employment Law for Managers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/employment-law-for-managers/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Employment Law", "HR Law", "Legal", "Management"],
                "description": "Course covering employment law for managers including HR law"
            },
            {
                "title": "Contract Law: From Trust to Promise to Contract",
                "provider": "edX",
                "url": "https://www.edx.org/course/contract-law-from-trust-to-promise-to-contract",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Contract Law", "Legal", "Contracts", "Business Law"],
                "description": "Course covering contract law including from trust to promise to contract"
            },
            {
                "title": "Business Law: Contract Law & Employment Law",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-law-contract-law/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Law", "Contract Law", "Employment Law", "Legal"],
                "description": "Course covering business law including contract law and employment law"
            },
            {
                "title": "CS50's Computer Science for Lawyers",
                "provider": "edX",
                "url": "https://www.edx.org/course/cs50s-computer-science-for-lawyers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Science", "Legal Tech", "Law", "Technology Law"],
                "description": "Course covering computer science for lawyers including legal tech"
            },
            {
                "title": "International Business Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/international-business-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["International Business Law", "Business Law", "International Law", "Legal"],
                "description": "Course covering international business law including international law"
            }
        ]
        
        log.info(f"📚 Adding {len(business_law_courses)} Business Law courses to knowledge base...")
        return self.batch_add_courses(business_law_courses)
    
    def add_data_privacy_gdpr_courses(self) -> bool:
        """
        Add Data Privacy & GDPR courses to knowledge base.
        Includes 10 courses covering GDPR compliance, data protection, privacy law, and CIPP certifications.
        
        Returns:
            True if successful, False otherwise
        """
        data_privacy_courses = [
            {
                "title": "Privacy Law and Data Protection",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/privacy-law-data-protection",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Privacy Law", "Data Protection", "GDPR", "Privacy"],
                "description": "Course covering privacy law and data protection including GDPR"
            },
            {
                "title": "Data Privacy Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/northeastern-data-privacy-fundamentals",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Privacy", "Privacy Fundamentals", "Data Protection"],
                "description": "Course covering data privacy fundamentals including data protection"
            },
            {
                "title": "GDPR Compliance: Essential Training",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/gdpr-certification/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GDPR", "GDPR Compliance", "Data Protection", "Privacy"],
                "description": "Course covering GDPR compliance including essential training"
            },
            {
                "title": "CIPP/E Cert Prep: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cipp-e-certification/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CIPP/E", "Privacy Certification", "GDPR", "Data Protection"],
                "description": "Course covering CIPP/E cert prep including the complete guide"
            },
            {
                "title": "Information Privacy Professionals (CIPP/US) Prep",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cipp-us-certification-prep/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CIPP/US", "Privacy Certification", "US Privacy Law", "Data Protection"],
                "description": "Course covering information privacy professionals CIPP/US prep"
            },
            {
                "title": "Understanding the GDPR",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gdpr",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GDPR", "Data Protection", "Privacy", "EU Law"],
                "description": "Course covering understanding the GDPR including EU law"
            },
            {
                "title": "Cybersecurity and Privacy in the IoT",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cybersecurity-privacy-iot",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cybersecurity", "Privacy", "IoT", "Data Protection"],
                "description": "Course covering cybersecurity and privacy in the IoT including data protection"
            },
            {
                "title": "Data Privacy Awareness",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/data-privacy-awareness/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Data Privacy", "Privacy Awareness", "Data Protection"],
                "description": "Course covering data privacy awareness including data protection"
            },
            {
                "title": "Healthcare Data Security, Privacy, and Compliance",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/healthcare-data-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Healthcare Data", "Data Security", "Privacy", "HIPAA", "Compliance"],
                "description": "Course covering healthcare data security, privacy, and compliance including HIPAA"
            },
            {
                "title": "Privacy and Security in Online Social Media",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/privacy-security-social-media",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Privacy", "Security", "Social Media", "Data Protection"],
                "description": "Course covering privacy and security in online social media including data protection"
            }
        ]
        
        log.info(f"📚 Adding {len(data_privacy_courses)} Data Privacy & GDPR courses to knowledge base...")
        return self.batch_add_courses(data_privacy_courses)
    
    def add_intellectual_property_courses(self) -> bool:
        """
        Add Intellectual Property (IP) courses to knowledge base.
        Includes 10 courses covering patents, copyrights, trademarks, and IP management.
        
        Returns:
            True if successful, False otherwise
        """
        ip_courses = [
            {
                "title": "Intellectual Property Law Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/intellectual-property-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intellectual Property", "IP Law", "Patents", "Copyright", "Trademarks"],
                "description": "Specialization covering intellectual property law including patents, copyright, and trademarks"
            },
            {
                "title": "General Course on Intellectual Property (DL-101)",
                "provider": "WIPO Academy",
                "url": "https://welc.wipo.int/acc/index.jsf?lang=en",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intellectual Property", "IP", "WIPO", "Patents", "Copyright"],
                "description": "Course covering general course on intellectual property including WIPO"
            },
            {
                "title": "Protecting Business Innovations via Patent",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/protecting-business-innovations-patent",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Patents", "Business Innovation", "IP", "Intellectual Property"],
                "description": "Course covering protecting business innovations via patent including IP"
            },
            {
                "title": "Copyright Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/copyright-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Copyright Law", "Copyright", "IP", "Intellectual Property"],
                "description": "Course covering copyright law including intellectual property"
            },
            {
                "title": "Trademark Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/trademark-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Trademark Law", "Trademarks", "IP", "Intellectual Property"],
                "description": "Course covering trademark law including intellectual property"
            },
            {
                "title": "Intellectual Property: Copyright, Trademarks & Patents",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/intellectual-property-rights/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intellectual Property", "Copyright", "Trademarks", "Patents"],
                "description": "Course covering intellectual property including copyright, trademarks and patents"
            },
            {
                "title": "Patents, Copyrights, and Trademarks for Entrepreneurs",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/patents-copyrights-trademarks/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Patents", "Copyright", "Trademarks", "Entrepreneurship", "IP"],
                "description": "Course covering patents, copyrights, and trademarks for entrepreneurs including IP"
            },
            {
                "title": "Patent Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/patent-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Patent Law", "Patents", "IP", "Intellectual Property"],
                "description": "Course covering patent law including intellectual property"
            },
            {
                "title": "Intellectual Property Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intellectual-property-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intellectual Property", "IP Management", "IP Strategy"],
                "description": "Course covering intellectual property management including IP strategy"
            },
            {
                "title": "Managing Intellectual Property in Universities",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/managing-intellectual-property-universities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intellectual Property", "IP Management", "University IP"],
                "description": "Course covering managing intellectual property in universities"
            }
        ]
        
        log.info(f"📚 Adding {len(ip_courses)} Intellectual Property (IP) courses to knowledge base...")
        return self.batch_add_courses(ip_courses)
    
    def add_cyber_law_ethics_courses(self) -> bool:
        """
        Add Cyber Law & Ethics courses to knowledge base.
        Includes 10 courses covering cybersecurity law, internet law, computer forensics, and AI ethics.
        
        Returns:
            True if successful, False otherwise
        """
        cyber_law_courses = [
            {
                "title": "Cybersecurity Law and Policy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cybersecurity-law-policy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cybersecurity Law", "Cyber Law", "Security Policy", "Legal"],
                "description": "Course covering cybersecurity law and policy including security policy"
            },
            {
                "title": "Cyber Law and Ethics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/cyber-law-and-ethics/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cyber Law", "Ethics", "Legal", "Cybersecurity"],
                "description": "Course covering cyber law and ethics including cybersecurity"
            },
            {
                "title": "Cloud Computing Law: Data Protection and Cybersecurity",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cloud-computing-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cloud Computing Law", "Data Protection", "Cybersecurity", "Legal"],
                "description": "Course covering cloud computing law including data protection and cybersecurity"
            },
            {
                "title": "Cyber Warfare and Terrorism",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cyber-warfare-terrorism",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cyber Warfare", "Cybersecurity", "Security", "Legal"],
                "description": "Course covering cyber warfare and terrorism including cybersecurity"
            },
            {
                "title": "Internet Law and Policy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/internet-law-policy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Internet Law", "Legal", "Policy", "Cyber Law"],
                "description": "Course covering internet law and policy including cyber law"
            },
            {
                "title": "Cybersecurity Ethics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cybersecurity-ethics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cybersecurity Ethics", "Ethics", "Cybersecurity", "Security"],
                "description": "Course covering cybersecurity ethics including security"
            },
            {
                "title": "Computer Forensics and Cyber Law",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/computer-forensics-cyber-law/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Computer Forensics", "Cyber Law", "Digital Forensics", "Legal"],
                "description": "Course covering computer forensics and cyber law including digital forensics"
            },
            {
                "title": "Legal Aspects of Cybersecurity",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/legal-aspects-of-cybersecurity/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cybersecurity", "Legal", "Security Law", "Compliance"],
                "description": "Course covering legal aspects of cybersecurity including security law"
            },
            {
                "title": "Ethics in the Age of AI",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ethics-in-age-of-ai",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI Ethics", "Ethics", "Artificial Intelligence", "Technology Ethics"],
                "description": "Course covering ethics in the age of AI including technology ethics"
            },
            {
                "title": "Surveillance Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/surveillance-law",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Surveillance Law", "Privacy Law", "Legal", "Security"],
                "description": "Course covering surveillance law including privacy law and security"
            }
        ]
        
        log.info(f"📚 Adding {len(cyber_law_courses)} Cyber Law & Ethics courses to knowledge base...")
        return self.batch_add_courses(cyber_law_courses)
    
    def add_corporate_governance_courses(self) -> bool:
        """
        Add Corporate Governance courses to knowledge base.
        Includes 10 courses covering corporate governance, business ethics, risk governance, and compliance.
        
        Returns:
            True if successful, False otherwise
        """
        corporate_governance_courses = [
            {
                "title": "Corporate Governance Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/corporate-governance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Governance", "Governance", "Business Ethics", "Compliance"],
                "description": "Specialization covering corporate governance including business ethics and compliance"
            },
            {
                "title": "Corporate Governance & Business Ethics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/corporate-governance-business-ethics/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Governance", "Business Ethics", "Ethics", "Governance"],
                "description": "Course covering corporate governance and business ethics"
            },
            {
                "title": "Board of Directors: Governance & Strategy",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/board-of-directors/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Board of Directors", "Governance", "Corporate Strategy", "Leadership"],
                "description": "Course covering board of directors including governance and strategy"
            },
            {
                "title": "Risk Governance: Manage the Risks",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/risk-governance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Risk Governance", "Risk Management", "Governance", "Compliance"],
                "description": "Course covering risk governance including managing the risks"
            },
            {
                "title": "Corporate Governance and Social Responsibility",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/corporate-governance-social-responsibility",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Governance", "Social Responsibility", "CSR", "Ethics"],
                "description": "Course covering corporate governance and social responsibility including CSR"
            },
            {
                "title": "Governance, Risk and Compliance (GRC)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/governance-risk-and-compliance-grc/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GRC", "Governance", "Risk Management", "Compliance"],
                "description": "Course covering governance, risk and compliance including GRC"
            },
            {
                "title": "Ethics and Social Responsibility",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ethics-social-responsibility",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Ethics", "Social Responsibility", "Business Ethics", "CSR"],
                "description": "Course covering ethics and social responsibility including business ethics"
            },
            {
                "title": "Corporate & Commercial Law II: Business Forms",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/corporate-commercial-law-2",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Law", "Commercial Law", "Business Forms", "Legal"],
                "description": "Course covering corporate and commercial law including business forms"
            },
            {
                "title": "Anti-Money Laundering (AML) Concepts",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/anti-money-laundering-aml-concepts/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AML", "Anti-Money Laundering", "Compliance", "Financial Crime"],
                "description": "Course covering anti-money laundering concepts including AML and compliance"
            },
            {
                "title": "Sarbanes-Oxley (SOX) Compliance",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sarbanes-oxley-sox-compliance/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["SOX", "Sarbanes-Oxley", "Compliance", "Financial Compliance"],
                "description": "Course covering Sarbanes-Oxley SOX compliance including financial compliance"
            }
        ]
        
        log.info(f"📚 Adding {len(corporate_governance_courses)} Corporate Governance courses to knowledge base...")
        return self.batch_add_courses(corporate_governance_courses)
    
    def add_public_health_courses(self) -> bool:
        """
        Add Public Health courses to knowledge base.
        Includes 10 courses covering epidemiology, global health, health policy, and public health practice.
        
        Returns:
            True if successful, False otherwise
        """
        public_health_courses = [
            {
                "title": "Epidemiology: The Basic Science of Public Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/epidemiology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Epidemiology", "Public Health", "Health Science", "Disease Prevention"],
                "description": "Course covering epidemiology including the basic science of public health"
            },
            {
                "title": "Foundations of Public Health Practice",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/public-health-practice",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Health", "Health Practice", "Health Policy", "Community Health"],
                "description": "Specialization covering foundations of public health practice including health policy"
            },
            {
                "title": "Global Health Security, Solidarity and Sustainability",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/global-health-security",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Health", "Health Security", "Public Health", "International Health"],
                "description": "Course covering global health security including solidarity and sustainability"
            },
            {
                "title": "Health Policy and the Affordable Care Act",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/health-policy-aca",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Health Policy", "Healthcare Policy", "ACA", "Public Policy"],
                "description": "Course covering health policy and the Affordable Care Act including healthcare policy"
            },
            {
                "title": "Systems Thinking in Public Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/systems-thinking-public-health",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Systems Thinking", "Public Health", "Health Systems", "Problem Solving"],
                "description": "Course covering systems thinking in public health including health systems"
            },
            {
                "title": "Public Health 101",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/public-health-101-course/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Health", "Health Basics", "Community Health"],
                "description": "Course covering public health 101 including health basics"
            },
            {
                "title": "An Introduction to the U.S. Food System",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/food-system",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Food System", "Nutrition", "Public Health", "Food Policy"],
                "description": "Course covering introduction to the U.S. food system including food policy"
            },
            {
                "title": "Essentials of Global Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/global-health",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Health", "International Health", "Public Health", "Health Systems"],
                "description": "Course covering essentials of global health including international health"
            },
            {
                "title": "Community Change in Public Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/community-change",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Community Health", "Public Health", "Community Change", "Health Promotion"],
                "description": "Course covering community change in public health including health promotion"
            },
            {
                "title": "Health Economics and Policy",
                "provider": "edX",
                "url": "https://www.edx.org/course/health-economics-and-policy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Health Economics", "Health Policy", "Economics", "Healthcare Policy"],
                "description": "Course covering health economics and policy including healthcare policy"
            }
        ]
        
        log.info(f"📚 Adding {len(public_health_courses)} Public Health courses to knowledge base...")
        return self.batch_add_courses(public_health_courses)
    
    def add_medical_coding_billing_courses(self) -> bool:
        """
        Add Medical Coding & Billing courses to knowledge base.
        Includes 10 courses covering medical coding, billing, ICD-10-CM, CPT, HCPCS, and revenue cycle management.
        
        Returns:
            True if successful, False otherwise
        """
        medical_coding_courses = [
            {
                "title": "Medical Billing and Coding Certification",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/medical-billing-and-coding-certification-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Billing", "Medical Coding", "Healthcare", "Certification"],
                "description": "Course covering medical billing and coding certification including healthcare"
            },
            {
                "title": "Medical Terminology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/medical-terminology",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Terminology", "Healthcare", "Medical Language"],
                "description": "Specialization covering medical terminology including medical language"
            },
            {
                "title": "CPC® Preparation Course (Official)",
                "provider": "AAPC",
                "url": "https://www.aapc.com/training/cpc-online-medical-coding-training-course.aspx",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CPC", "Medical Coding", "Certification", "AAPC"],
                "description": "Course covering CPC preparation course including official AAPC certification"
            },
            {
                "title": "Medical Coding: ICD-10-CM",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/medical-coding-icd-10-cm-essentials/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Coding", "ICD-10-CM", "Diagnosis Coding", "Healthcare"],
                "description": "Course covering medical coding ICD-10-CM including diagnosis coding"
            },
            {
                "title": "Revenue Cycle Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/revenue-cycle-management-healthcare",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Revenue Cycle Management", "Healthcare Finance", "Medical Billing", "Healthcare"],
                "description": "Course covering revenue cycle management including healthcare finance"
            },
            {
                "title": "Medical Coding: CPT & HCPCS",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/medical-coding-cpt-hcpcs-essentials/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Coding", "CPT", "HCPCS", "Procedure Coding", "Healthcare"],
                "description": "Course covering medical coding CPT and HCPCS including procedure coding"
            },
            {
                "title": "Introduction to Medical Coding",
                "provider": "Alison",
                "url": "https://alison.com/course/introduction-to-medical-coding-revised",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Coding", "Healthcare", "Introduction"],
                "description": "Course covering introduction to medical coding including healthcare"
            },
            {
                "title": "Become a Medical Biller and Coder",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/paths/become-a-medical-biller-and-coder",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Billing", "Medical Coding", "Healthcare", "Career"],
                "description": "Learning path covering become a medical biller and coder including career"
            },
            {
                "title": "Crash Course in Medical Billing & Coding",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/crash-course-in-medical-billing-coding/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Billing", "Medical Coding", "Healthcare", "Basics"],
                "description": "Course covering crash course in medical billing and coding including basics"
            },
            {
                "title": "Healthcare Marketplace",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/healthcare-marketplace",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Healthcare Marketplace", "Healthcare Economics", "Health Policy"],
                "description": "Course covering healthcare marketplace including healthcare economics"
            }
        ]
        
        log.info(f"📚 Adding {len(medical_coding_courses)} Medical Coding & Billing courses to knowledge base...")
        return self.batch_add_courses(medical_coding_courses)
    
    def add_psychology_behavioral_science_courses(self) -> bool:
        """
        Add Psychology & Behavioral Science courses to knowledge base.
        Includes 10 courses covering psychology, behavioral economics, social psychology, and neuroscience.
        
        Returns:
            True if successful, False otherwise
        """
        psychology_courses = [
            {
                "title": "Introduction to Psychology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-psychology",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Psychology", "Introduction to Psychology", "Behavioral Science"],
                "description": "Course covering introduction to psychology including behavioral science"
            },
            {
                "title": "The Science of Well-Being",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/the-science-of-well-being",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Well-Being", "Positive Psychology", "Happiness", "Mental Health"],
                "description": "Course covering the science of well-being including positive psychology"
            },
            {
                "title": "Social Psychology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/social-psychology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Social Psychology", "Psychology", "Behavioral Science", "Social Behavior"],
                "description": "Course covering social psychology including social behavior"
            },
            {
                "title": "Introduction to Human Behavioral Genetics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/behavioralgenetics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Behavioral Genetics", "Psychology", "Genetics", "Behavioral Science"],
                "description": "Course covering introduction to human behavioral genetics including behavioral science"
            },
            {
                "title": "Behavioral Economics in Action",
                "provider": "edX",
                "url": "https://www.edx.org/course/behavioral-economics-in-action",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Behavioral Economics", "Economics", "Psychology", "Decision Making"],
                "description": "Course covering behavioral economics in action including decision making"
            },
            {
                "title": "Positive Psychology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/positive-psychology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Positive Psychology", "Psychology", "Well-Being", "Happiness"],
                "description": "Course covering positive psychology including well-being and happiness"
            },
            {
                "title": "Abnormal Psychology",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/abnormal-psychology-psychological-disorders/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Abnormal Psychology", "Psychology", "Mental Health", "Psychological Disorders"],
                "description": "Course covering abnormal psychology including psychological disorders"
            },
            {
                "title": "Moralities of Everyday Life",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/moralities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Moral Psychology", "Psychology", "Ethics", "Social Psychology"],
                "description": "Course covering moralities of everyday life including moral psychology"
            },
            {
                "title": "Understanding the Brain: The Neurobiology of Everyday Life",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/neurobiology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Neurobiology", "Neuroscience", "Brain Science", "Psychology"],
                "description": "Course covering understanding the brain including neurobiology and neuroscience"
            },
            {
                "title": "Foundations of Psychology",
                "provider": "edX",
                "url": "https://www.edx.org/course/foundations-of-psychology",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Psychology", "Foundations", "Behavioral Science", "Psychology Basics"],
                "description": "Course covering foundations of psychology including psychology basics"
            }
        ]
        
        log.info(f"📚 Adding {len(psychology_courses)} Psychology & Behavioral Science courses to knowledge base...")
        return self.batch_add_courses(psychology_courses)
    
    def add_nutrition_wellness_courses(self) -> bool:
        """
        Add Nutrition & Wellness courses to knowledge base.
        Includes 10 courses covering nutrition, food and health, exercise science, and wellness coaching.
        
        Returns:
            True if successful, False otherwise
        """
        nutrition_courses = [
            {
                "title": "Stanford Introduction to Food and Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/food-and-health",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Nutrition", "Food and Health", "Wellness", "Healthy Eating"],
                "description": "Course covering Stanford introduction to food and health including healthy eating"
            },
            {
                "title": "Nutrition and Health: Macronutrients",
                "provider": "edX",
                "url": "https://www.edx.org/course/nutrition-and-health-macronutrients-and-overnutrition",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Nutrition", "Macronutrients", "Health", "Nutrition Science"],
                "description": "Course covering nutrition and health including macronutrients and nutrition science"
            },
            {
                "title": "Science of Exercise",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/science-of-exercise",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Exercise Science", "Fitness", "Health", "Physical Activity"],
                "description": "Course covering science of exercise including fitness and physical activity"
            },
            {
                "title": "Child Nutrition and Cooking",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/childnutrition",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Child Nutrition", "Nutrition", "Cooking", "Pediatric Nutrition"],
                "description": "Course covering child nutrition and cooking including pediatric nutrition"
            },
            {
                "title": "Internationally Accredited Diploma Certificate in Nutrition",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/internationally-accredited-diploma-certificate-in-nutrition/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Nutrition", "Nutrition Certification", "Health", "Wellness"],
                "description": "Course covering internationally accredited diploma certificate in nutrition"
            },
            {
                "title": "Weight Management: Beyond Balancing Calories",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/weight-management-beyond-balancing-calories",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Weight Management", "Nutrition", "Health", "Wellness"],
                "description": "Course covering weight management including beyond balancing calories"
            },
            {
                "title": "Health and Wellness Coaching",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/health-and-wellness-coaching",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Wellness Coaching", "Health Coaching", "Wellness", "Health"],
                "description": "Specialization covering health and wellness coaching including health coaching"
            },
            {
                "title": "Gut Check: Exploring Your Microbiome",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/microbiome",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Microbiome", "Gut Health", "Nutrition", "Health"],
                "description": "Course covering gut check including exploring your microbiome and gut health"
            },
            {
                "title": "The Science of Gastronomy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gastronomy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Gastronomy", "Food Science", "Cooking", "Nutrition"],
                "description": "Course covering the science of gastronomy including food science and cooking"
            },
            {
                "title": "Nutrition and Lifestyle in Pregnancy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/nutrition-pregnancy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pregnancy Nutrition", "Nutrition", "Maternal Health", "Health"],
                "description": "Course covering nutrition and lifestyle in pregnancy including maternal health"
            }
        ]
        
        log.info(f"📚 Adding {len(nutrition_courses)} Nutrition & Wellness courses to knowledge base...")
        return self.batch_add_courses(nutrition_courses)
    
    def add_mental_health_first_aid_courses(self) -> bool:
        """
        Add Mental Health First Aid courses to knowledge base.
        Includes 10 courses covering psychological first aid, mental health awareness, stress management, and addiction treatment.
        
        Returns:
            True if successful, False otherwise
        """
        mental_health_courses = [
            {
                "title": "Psychological First Aid",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/psychological-first-aid",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Psychological First Aid", "Mental Health", "Crisis Intervention", "Support"],
                "description": "Course covering psychological first aid including crisis intervention and support"
            },
            {
                "title": "Managing Mental Health and Stress",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/managing-mental-health-and-stress",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mental Health", "Stress Management", "Wellness", "Mental Wellness"],
                "description": "Course covering managing mental health and stress including mental wellness"
            },
            {
                "title": "Mental Health First Aid Certification",
                "provider": "National Council",
                "url": "https://www.mentalhealthfirstaid.org/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mental Health First Aid", "Certification", "Mental Health", "First Aid"],
                "description": "Course covering mental health first aid certification including first aid"
            },
            {
                "title": "The Social Context of Mental Health and Illness",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mental-health",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mental Health", "Social Psychology", "Mental Illness", "Public Health"],
                "description": "Course covering the social context of mental health and illness including public health"
            },
            {
                "title": "Addiction Treatment: Clinical Skills",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/addiction-treatment",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Addiction Treatment", "Substance Abuse", "Mental Health", "Clinical Skills"],
                "description": "Course covering addiction treatment including clinical skills and substance abuse"
            },
            {
                "title": "Talk to Me: Improving Mental Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/talk-to-me",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mental Health", "Communication", "Mental Wellness", "Support"],
                "description": "Course covering talk to me including improving mental health and support"
            },
            {
                "title": "Mindfulness and Resilience to Stress at Work",
                "provider": "edX",
                "url": "https://www.edx.org/course/mindfulness-and-resilience-to-stress-at-work",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mindfulness", "Stress Management", "Resilience", "Workplace Wellness"],
                "description": "Course covering mindfulness and resilience to stress at work including workplace wellness"
            },
            {
                "title": "Mental Health Ambassador Certificate",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mental-health-ambassador-certificate/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mental Health", "Mental Health Ambassador", "Certification", "Advocacy"],
                "description": "Course covering mental health ambassador certificate including advocacy"
            },
            {
                "title": "Caring for Others",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/caring-for-others",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Caregiving", "Mental Health", "Support", "Compassion"],
                "description": "Course covering caring for others including support and compassion"
            },
            {
                "title": "Positive Psychiatry and Mental Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/positive-psychiatry",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Positive Psychiatry", "Mental Health", "Psychiatry", "Well-Being"],
                "description": "Course covering positive psychiatry and mental health including well-being"
            }
        ]
        
        log.info(f"📚 Adding {len(mental_health_courses)} Mental Health First Aid courses to knowledge base...")
        return self.batch_add_courses(mental_health_courses)
    
    def add_business_english_courses(self) -> bool:
        """
        Add Business English courses to knowledge base.
        Includes 10 courses covering business English communication, professional emails, networking, and entrepreneurship.
        
        Returns:
            True if successful, False otherwise
        """
        business_english_courses = [
            {
                "title": "Business English Communication Skills Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/business-english",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "English Communication", "Professional English", "Business Communication"],
                "description": "Specialization covering business English communication skills including professional English"
            },
            {
                "title": "English for Career Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/careerdevelopment",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["English", "Career Development", "Professional English", "Business English"],
                "description": "Course covering English for career development including professional and business English"
            },
            {
                "title": "Business English for Non-Native Speakers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/business-english-for-non-native-speakers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "ESL", "English for Non-Native Speakers", "Professional English"],
                "description": "Specialization covering business English for non-native speakers including ESL"
            },
            {
                "title": "Business English Course for ESL Students",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-english-course-for-esl-students/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "ESL", "English for ESL", "Professional English"],
                "description": "Course covering business English course for ESL students including professional English"
            },
            {
                "title": "Write Professional Emails in English",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/professional-emails-english",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Email Writing", "Professional Writing", "Business Communication"],
                "description": "Course covering writing professional emails in English including business communication"
            },
            {
                "title": "Business English: Networking",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/business-english-networking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Networking", "Professional English", "Business Communication"],
                "description": "Course covering business English networking including professional English"
            },
            {
                "title": "English for Business and Entrepreneurship",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/business-entrepreneurship-english",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Entrepreneurship", "English for Business", "Professional English"],
                "description": "Course covering English for business and entrepreneurship including professional English"
            },
            {
                "title": "Advanced Business English Communication",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/advanced-business-english-vocabulary/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Advanced English", "Business Vocabulary", "Professional English"],
                "description": "Course covering advanced business English communication including business vocabulary"
            },
            {
                "title": "Speak English Professionally: In Person, Online & On the Phone",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/speak-english-professionally",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Speaking English", "Professional Speaking", "Business Communication"],
                "description": "Course covering speaking English professionally including in person, online and on the phone"
            },
            {
                "title": "Essential Business English",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/essential-business-english/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business English", "Essential English", "Professional English", "Business Basics"],
                "description": "Course covering essential business English including business basics"
            }
        ]
        
        log.info(f"📚 Adding {len(business_english_courses)} Business English courses to knowledge base...")
        return self.batch_add_courses(business_english_courses)
    
    def add_spanish_business_courses(self) -> bool:
        """
        Add Spanish (Business Focus) courses to knowledge base.
        Includes 10 courses covering business Spanish, professional communication, healthcare Spanish, and Spanish for business professionals.
        
        Returns:
            True if successful, False otherwise
        """
        spanish_business_courses = [
            {
                "title": "Business Spanish Certificate Program",
                "provider": "UW-Madison",
                "url": "https://continuingeducation.wisc.edu/courses/business-spanish-certificate-program/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Business Spanish", "Spanish for Business", "Professional Spanish"],
                "description": "Course covering business Spanish certificate program including professional Spanish"
            },
            {
                "title": "Spanish for Business: Professional Communication",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/spanish-for-business/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Business Spanish", "Professional Communication", "Spanish for Business"],
                "description": "Course covering Spanish for business including professional communication"
            },
            {
                "title": "Learn Spanish: Basic Spanish Vocabulary (Business Context)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/learn-spanish",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Spanish Vocabulary", "Business Spanish", "Basic Spanish"],
                "description": "Specialization covering learn Spanish including basic Spanish vocabulary in business context"
            },
            {
                "title": "Spanish for Business Professionals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/spanish-for-business-professionals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Business Spanish", "Professional Spanish", "Spanish for Professionals"],
                "description": "Course covering Spanish for business professionals including professional Spanish"
            },
            {
                "title": "Professional Spanish Certificate",
                "provider": "edX",
                "url": "https://www.edx.org/certificates/professional-certificate/asu-professional-spanish",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Professional Spanish", "Spanish Certificate", "Business Spanish"],
                "description": "Course covering professional Spanish certificate including business Spanish"
            },
            {
                "title": "Basic Spanish 1: Getting Started",
                "provider": "edX",
                "url": "https://www.edx.org/course/basic-spanish-1-getting-started",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Basic Spanish", "Spanish for Beginners", "Spanish Language"],
                "description": "Course covering basic Spanish 1 including getting started with Spanish language"
            },
            {
                "title": "Spanish for Healthcare",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/spanish-for-healthcare",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Healthcare Spanish", "Medical Spanish", "Spanish for Healthcare"],
                "description": "Course covering Spanish for healthcare including medical Spanish"
            },
            {
                "title": "Intermediate Spanish: Business & Culture",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/intermediate-spanish-business-culture/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Business Spanish", "Spanish Culture", "Intermediate Spanish"],
                "description": "Course covering intermediate Spanish including business and culture"
            },
            {
                "title": "Spanish for Successful Communication in Healthcare",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/spanish-successful-communication-healthcare",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Healthcare Spanish", "Medical Spanish", "Spanish Communication"],
                "description": "Course covering Spanish for successful communication in healthcare including medical Spanish"
            },
            {
                "title": "Spanish Language: Levels 1 & 2",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/el-metodo-spanish-levels-1-2/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Spanish", "Spanish Language", "Basic Spanish", "Spanish for Beginners"],
                "description": "Course covering Spanish language levels 1 and 2 including basic Spanish"
            }
        ]
        
        log.info(f"📚 Adding {len(spanish_business_courses)} Spanish (Business Focus) courses to knowledge base...")
        return self.batch_add_courses(spanish_business_courses)
    
    def add_mandarin_german_french_courses(self) -> bool:
        """
        Add Mandarin, German & French (Business Focus) courses to knowledge base.
        Includes 10 courses covering business Chinese, business German, business French, and professional language skills.
        
        Returns:
            True if successful, False otherwise
        """
        multilingual_business_courses = [
            {
                "title": "Business Chinese: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-chinese/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Chinese", "Mandarin", "Business Chinese", "Chinese for Business"],
                "description": "Course covering business Chinese including the complete guide to Chinese for business"
            },
            {
                "title": "Mandarin Chinese for Business",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mandarin-chinese-business",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mandarin", "Chinese", "Business Chinese", "Mandarin for Business"],
                "description": "Course covering Mandarin Chinese for business including business Chinese"
            },
            {
                "title": "Business German: A Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-german/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["German", "Business German", "German for Business", "Professional German"],
                "description": "Course covering business German including a complete course on German for business"
            },
            {
                "title": "German at Work",
                "provider": "Goethe-Institut",
                "url": "https://www.goethe.de/en/spr/kur/b2b.html",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["German", "Business German", "German at Work", "Professional German"],
                "description": "Course covering German at work including business and professional German"
            },
            {
                "title": "Business French: French for Professionals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-french/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["French", "Business French", "French for Professionals", "Professional French"],
                "description": "Course covering business French including French for professionals"
            },
            {
                "title": "French for Professional Contexts",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/etudier-en-france",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["French", "Professional French", "French for Business", "Business French"],
                "description": "Course covering French for professional contexts including business French"
            },
            {
                "title": "Chinese for HSK 4 (Business Level)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/hsk-4",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Chinese", "Mandarin", "HSK 4", "Business Chinese"],
                "description": "Course covering Chinese for HSK 4 including business level Chinese"
            },
            {
                "title": "Mastering German: Business & Professional",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-german-business-professional/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["German", "Business German", "Professional German", "Advanced German"],
                "description": "Course covering mastering German including business and professional German"
            },
            {
                "title": "French for Business: The Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/french-for-business/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["French", "Business French", "French for Business", "Professional French"],
                "description": "Course covering French for business including the complete course"
            },
            {
                "title": "Basic Mandarin Chinese for Business",
                "provider": "edX",
                "url": "https://www.edx.org/course/basic-mandarin-chinese-level-1",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mandarin", "Chinese", "Basic Chinese", "Business Chinese"],
                "description": "Course covering basic Mandarin Chinese for business including basic Chinese"
            }
        ]
        
        log.info(f"📚 Adding {len(multilingual_business_courses)} Mandarin, German & French (Business Focus) courses to knowledge base...")
        return self.batch_add_courses(multilingual_business_courses)
    
    def add_cross_cultural_communication_courses(self) -> bool:
        """
        Add Cross-Cultural Communication courses to knowledge base.
        Includes 10 courses covering intercultural communication, global teams, cultural management, and business etiquette.
        
        Returns:
            True if successful, False otherwise
        """
        cross_cultural_courses = [
            {
                "title": "Cross-Cultural Communication and Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/cross-cultural-communication-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cross-Cultural Communication", "Intercultural Communication", "Cultural Management", "Global Business"],
                "description": "Course covering cross-cultural communication and management including intercultural communication"
            },
            {
                "title": "Intercultural Communication and Conflict Resolution",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intercultural-communication",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Intercultural Communication", "Cross-Cultural Communication", "Conflict Resolution", "Cultural Communication"],
                "description": "Course covering intercultural communication and conflict resolution including cultural communication"
            },
            {
                "title": "Working in a Global Team",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/working-in-a-global-team",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Teams", "Cross-Cultural Communication", "Teamwork", "International Business"],
                "description": "Course covering working in a global team including cross-cultural communication"
            },
            {
                "title": "Global Impact: Cross-Cultural Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/global-impact-cross-cultural-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cross-Cultural Management", "Global Business", "Cultural Management", "International Management"],
                "description": "Course covering global impact including cross-cultural management"
            },
            {
                "title": "Communicating Across Cultures",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/communicating-across-cultures",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Cross-Cultural Communication", "Cultural Communication", "Intercultural Communication", "Communication"],
                "description": "Course covering communicating across cultures including intercultural communication"
            },
            {
                "title": "International Business Culture and Etiquette",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/international-business-culture-and-etiquette/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Etiquette", "International Business", "Cultural Etiquette", "Cross-Cultural Business"],
                "description": "Course covering international business culture and etiquette including cross-cultural business"
            },
            {
                "title": "Leading Diverse Teams & Organizations",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/leading-diverse-teams",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Diversity", "Team Leadership", "Cross-Cultural Management", "Inclusive Leadership"],
                "description": "Course covering leading diverse teams and organizations including cross-cultural management"
            },
            {
                "title": "Culture and Globalization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/culture-globalization",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Culture", "Globalization", "Cultural Studies", "International Business"],
                "description": "Course covering culture and globalization including cultural studies"
            },
            {
                "title": "Business Etiquette: Master Communication & Soft Skills",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/business-etiquette-101-social-skills-for-success/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Etiquette", "Communication Skills", "Soft Skills", "Professional Etiquette"],
                "description": "Course covering business etiquette including mastering communication and soft skills"
            },
            {
                "title": "Communication in the Global Workplace",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/communication-global-workplace",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Communication", "Workplace Communication", "Cross-Cultural Communication", "International Business"],
                "description": "Course covering communication in the global workplace including cross-cultural communication"
            }
        ]
        
        log.info(f"📚 Adding {len(cross_cultural_courses)} Cross-Cultural Communication courses to knowledge base...")
        return self.batch_add_courses(cross_cultural_courses)
    
    def add_translation_localization_courses(self) -> bool:
        """
        Add Translation & Localization courses to knowledge base.
        Includes 10 courses covering translation practice, localization, machine translation, and translation quality management.
        
        Returns:
            True if successful, False otherwise
        """
        translation_courses = [
            {
                "title": "Translation in Practice",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/translation-in-practice",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Translation", "Translation Practice", "Language Translation", "Professional Translation"],
                "description": "Course covering translation in practice including professional translation"
            },
            {
                "title": "Localization: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/localization-the-complete-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Localization", "Translation", "Internationalization", "Globalization"],
                "description": "Course covering localization including the complete guide to translation and internationalization"
            },
            {
                "title": "Website Localization for Translators",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/website-localization-for-translators/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Localization", "Website Localization", "Translation", "Web Translation"],
                "description": "Course covering website localization for translators including web translation"
            },
            {
                "title": "Machine Translation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/machine-translation",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Machine Translation", "Translation Technology", "NLP", "AI Translation"],
                "description": "Course covering machine translation including translation technology and AI translation"
            },
            {
                "title": "International Organizations Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/international-organizations",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["International Organizations", "Global Management", "Translation", "Multilingual Management"],
                "description": "Course covering international organizations management including multilingual management"
            },
            {
                "title": "Introduction to Localization",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/introduction-to-localization/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Localization", "Translation", "Introduction", "Localization Basics"],
                "description": "Course covering introduction to localization including localization basics"
            },
            {
                "title": "Working as a Translator",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/working-as-a-translator/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Translation", "Translator Career", "Professional Translation", "Translation Career"],
                "description": "Course covering working as a translator including translator career"
            },
            {
                "title": "Translation Quality Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/translation-quality-management/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Translation", "Quality Management", "Translation Quality", "Translation Standards"],
                "description": "Course covering translation quality management including translation standards"
            },
            {
                "title": "Software Localization for Translators",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/software-localization-for-translators/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Localization", "Software Localization", "Translation", "Software Translation"],
                "description": "Course covering software localization for translators including software translation"
            },
            {
                "title": "Audiovisual Translation: Subtitling",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/audiovisual-translation-subtitling/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Audiovisual Translation", "Subtitling", "Translation", "Video Translation"],
                "description": "Course covering audiovisual translation including subtitling and video translation"
            }
        ]
        
        log.info(f"📚 Adding {len(translation_courses)} Translation & Localization courses to knowledge base...")
        return self.batch_add_courses(translation_courses)
    
    def add_linear_algebra_courses(self) -> bool:
        """
        Add Linear Algebra courses to knowledge base.
        Includes 10 courses covering linear algebra for machine learning, data science, matrix algebra, and eigenvectors.
        
        Returns:
            True if successful, False otherwise
        """
        linear_algebra_courses = [
            {
                "title": "Mathematics for Machine Learning: Linear Algebra",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/linear-algebra-machine-learning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Machine Learning", "Mathematics", "Data Science"],
                "description": "Course covering mathematics for machine learning including linear algebra"
            },
            {
                "title": "Linear Algebra for Machine Learning and Data Science",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/machine-learning-linear-algebra",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Machine Learning", "Data Science", "Mathematics"],
                "description": "Course covering linear algebra for machine learning and data science"
            },
            {
                "title": "Introduction to Linear Algebra",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-linear-algebra",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Mathematics", "Introduction", "Algebra"],
                "description": "Course covering introduction to linear algebra including mathematics"
            },
            {
                "title": "Matrix Algebra for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/matrix-algebra-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Matrix Algebra", "Linear Algebra", "Engineering Mathematics", "Mathematics"],
                "description": "Course covering matrix algebra for engineers including engineering mathematics"
            },
            {
                "title": "Linear Algebra from Elementary to Advanced",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/linear-algebra-elementary-to-advanced",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Mathematics", "Advanced Mathematics"],
                "description": "Specialization covering linear algebra from elementary to advanced"
            },
            {
                "title": "Complete Linear Algebra for Data Science",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/linear-algebra-for-data-science-machine-learning-ai/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Data Science", "Machine Learning", "Mathematics"],
                "description": "Course covering complete linear algebra for data science including machine learning"
            },
            {
                "title": "First Steps in Linear Algebra for Machine Learning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/linear-algebra-machine-learning-first-steps",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Machine Learning", "Mathematics", "Introduction"],
                "description": "Course covering first steps in linear algebra for machine learning"
            },
            {
                "title": "Linear Algebra and Its Applications",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/linear-algebra-and-its-applications/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Mathematics", "Applications"],
                "description": "Course covering linear algebra and its applications"
            },
            {
                "title": "Master Linear Algebra",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/master-linear-algebra/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Mathematics", "Master Linear Algebra"],
                "description": "Course covering master linear algebra including mathematics"
            },
            {
                "title": "Linear Algebra: Matrix Algebra, Determinants, & Eigenvectors",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/matrix-algebra-determinants-eigenvectors",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Linear Algebra", "Matrix Algebra", "Determinants", "Eigenvectors"],
                "description": "Course covering linear algebra including matrix algebra, determinants and eigenvectors"
            }
        ]
        
        log.info(f"📚 Adding {len(linear_algebra_courses)} Linear Algebra courses to knowledge base...")
        return self.batch_add_courses(linear_algebra_courses)
    
    def add_calculus_engineers_courses(self) -> bool:
        """
        Add Calculus for Engineers courses to knowledge base.
        Includes 10 courses covering calculus, vector calculus, differential calculus, and multivariable calculus.
        
        Returns:
            True if successful, False otherwise
        """
        calculus_courses = [
            {
                "title": "Introduction to Calculus",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-calculus",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Mathematics", "Introduction", "Math"],
                "description": "Course covering introduction to calculus including mathematics"
            },
            {
                "title": "Calculus: Single Variable Part 1 - Functions",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/single-variable-calculus",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Single Variable Calculus", "Functions", "Mathematics"],
                "description": "Course covering calculus single variable part 1 including functions"
            },
            {
                "title": "Vector Calculus for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/vector-calculus-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Vector Calculus", "Calculus", "Engineering Mathematics", "Mathematics"],
                "description": "Course covering vector calculus for engineers including engineering mathematics"
            },
            {
                "title": "Calculus for Machine Learning and Data Science",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/calculus-for-machine-learning-and-data-science",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Machine Learning", "Data Science", "Mathematics"],
                "description": "Course covering calculus for machine learning and data science"
            },
            {
                "title": "Mathematics for Engineers: The Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/mathematics-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mathematics", "Engineering Mathematics", "Calculus", "Math for Engineers"],
                "description": "Specialization covering mathematics for engineers including calculus"
            },
            {
                "title": "Become a Calculus 1 Master",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/calculus1/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Calculus 1", "Mathematics", "Math"],
                "description": "Course covering become a calculus 1 master including mathematics"
            },
            {
                "title": "Calculus for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/calculus-for-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Engineering Mathematics", "Mathematics", "Math for Engineers"],
                "description": "Course covering calculus for engineers including engineering mathematics"
            },
            {
                "title": "Pre-University Calculus",
                "provider": "edX",
                "url": "https://www.edx.org/course/pre-university-calculus",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Calculus", "Pre-University", "Mathematics", "Math"],
                "description": "Course covering pre-university calculus including mathematics"
            },
            {
                "title": "Differential Calculus through Data and Modeling",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/differential-calculus-data-modeling",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Differential Calculus", "Calculus", "Data Modeling", "Mathematics"],
                "description": "Specialization covering differential calculus through data and modeling"
            },
            {
                "title": "Multivariable Calculus",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/multivariable-calculus-2/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Multivariable Calculus", "Calculus", "Mathematics", "Advanced Calculus"],
                "description": "Course covering multivariable calculus including advanced calculus"
            }
        ]
        
        log.info(f"📚 Adding {len(calculus_courses)} Calculus for Engineers courses to knowledge base...")
        return self.batch_add_courses(calculus_courses)
    
    def add_probability_statistics_courses(self) -> bool:
        """
        Add Probability & Statistics courses to knowledge base.
        Includes 10 courses covering probability, statistics, Bayesian statistics, and statistical analysis.
        
        Returns:
            True if successful, False otherwise
        """
        probability_statistics_courses = [
            {
                "title": "Probability and Statistics: To p or not to p?",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/probability-statistics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "Statistics", "Statistical Analysis", "Mathematics"],
                "description": "Course covering probability and statistics including statistical analysis"
            },
            {
                "title": "Statistics with Python Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/statistics-with-python",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statistics", "Python", "Statistical Analysis", "Data Analysis"],
                "description": "Specialization covering statistics with Python including statistical analysis"
            },
            {
                "title": "Introduction to Probability and Data with R",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/probability-intro",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "Statistics", "R Programming", "Data Analysis"],
                "description": "Course covering introduction to probability and data with R including data analysis"
            },
            {
                "title": "Probability - The Science of Uncertainty and Data",
                "provider": "edX",
                "url": "https://www.edx.org/course/probability-the-science-of-uncertainty-and-data",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "Statistics", "Data Science", "Mathematics"],
                "description": "Course covering probability including the science of uncertainty and data"
            },
            {
                "title": "Statistics for Data Science and Business Analysis",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/statistics-for-data-science-and-business-analysis/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statistics", "Data Science", "Business Analysis", "Statistical Analysis"],
                "description": "Course covering statistics for data science and business analysis"
            },
            {
                "title": "Bayesian Statistics: From Concept to Data Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bayesian-statistics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bayesian Statistics", "Statistics", "Statistical Analysis", "Data Analysis"],
                "description": "Course covering Bayesian statistics including from concept to data analysis"
            },
            {
                "title": "Introduction to Statistics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/stanford-statistics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statistics", "Introduction", "Statistical Analysis", "Mathematics"],
                "description": "Course covering introduction to statistics including statistical analysis"
            },
            {
                "title": "Probability & Statistics for Data Science & Business",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/probability-and-statistics-for-business-and-data-science/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "Statistics", "Data Science", "Business Statistics"],
                "description": "Course covering probability and statistics for data science and business"
            },
            {
                "title": "Foundations of Probability in R",
                "provider": "DataCamp",
                "url": "https://www.datacamp.com/courses/foundations-of-probability-in-r",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Probability", "R Programming", "Statistics", "Data Analysis"],
                "description": "Course covering foundations of probability in R including statistics"
            },
            {
                "title": "Fundamentals of Statistics",
                "provider": "edX",
                "url": "https://www.edx.org/course/fundamentals-of-statistics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statistics", "Fundamentals", "Statistical Analysis", "Mathematics"],
                "description": "Course covering fundamentals of statistics including statistical analysis"
            }
        ]
        
        log.info(f"📚 Adding {len(probability_statistics_courses)} Probability & Statistics courses to knowledge base...")
        return self.batch_add_courses(probability_statistics_courses)
    
    def add_research_methods_courses(self) -> bool:
        """
        Add Research Methods (Academic & UX) courses to knowledge base.
        Includes 10 courses covering quantitative methods, qualitative research, UX research, and market research.
        
        Returns:
            True if successful, False otherwise
        """
        research_methods_courses = [
            {
                "title": "Quantitative Methods",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/quantitative-methods",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Quantitative Methods", "Research Methods", "Data Analysis", "Statistics"],
                "description": "Course covering quantitative methods including research methods and data analysis"
            },
            {
                "title": "Qualitative Research Methods",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/qualitative-research",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Qualitative Research", "Research Methods", "Research", "Data Collection"],
                "description": "Course covering qualitative research methods including research and data collection"
            },
            {
                "title": "Understanding Research Methods",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/research-methods",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Research Methods", "Research", "Academic Research", "Research Design"],
                "description": "Course covering understanding research methods including research design"
            },
            {
                "title": "UX Research at Scale: Surveys, Analytics, Online Testing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/ux-research-at-scale",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Research", "User Research", "Research Methods", "User Experience"],
                "description": "Course covering UX research at scale including surveys, analytics and online testing"
            },
            {
                "title": "Conduct UX Research and Test Early Concepts",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/conduct-ux-research",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Research", "User Research", "User Testing", "User Experience"],
                "description": "Course covering conducting UX research and testing early concepts"
            },
            {
                "title": "User Experience Research and Design Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/michiganux",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Research", "User Experience", "UX Design", "User Research"],
                "description": "Specialization covering user experience research and design including UX research"
            },
            {
                "title": "Market Research Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/market-research",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Market Research", "Research Methods", "Business Research", "Consumer Research"],
                "description": "Specialization covering market research including business and consumer research"
            },
            {
                "title": "Research Methods for Business Students",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/research-methods-for-business-students/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Research Methods", "Business Research", "Academic Research", "Research"],
                "description": "Course covering research methods for business students including business research"
            },
            {
                "title": "Tech for Good: The Role of Research and Design",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/tech-for-good-research-and-design",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Research", "Design", "Tech for Good", "Research Methods"],
                "description": "Course covering tech for good including the role of research and design"
            },
            {
                "title": "UX Research Methods: Interviewing",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/ux-research-methods-interviewing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UX Research", "User Interviewing", "Research Methods", "User Research"],
                "description": "Course covering UX research methods including interviewing"
            }
        ]
        
        log.info(f"📚 Adding {len(research_methods_courses)} Research Methods (Academic & UX) courses to knowledge base...")
        return self.batch_add_courses(research_methods_courses)
    
    def add_game_theory_courses(self) -> bool:
        """
        Add Game Theory courses to knowledge base.
        Includes 10 courses covering game theory, strategic thinking, competitive strategy, and mathematical game theory.
        
        Returns:
            True if successful, False otherwise
        """
        game_theory_courses = [
            {
                "title": "Game Theory",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/game-theory-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Strategic Thinking", "Economics", "Decision Making"],
                "description": "Course covering game theory including strategic thinking and decision making"
            },
            {
                "title": "Game Theory II: Advanced Applications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/game-theory-2",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Advanced Game Theory", "Strategic Thinking", "Economics"],
                "description": "Course covering game theory II including advanced applications"
            },
            {
                "title": "Welcome to Game Theory",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/game-theory-introduction",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Introduction", "Strategic Thinking", "Economics"],
                "description": "Course covering welcome to game theory including introduction"
            },
            {
                "title": "Strategy and Game Theory for Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/strategy-and-game-theory",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Strategy", "Management", "Strategic Thinking"],
                "description": "Course covering strategy and game theory for management including strategic thinking"
            },
            {
                "title": "Games without Chance: Combinatorial Game Theory",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/combinatorial-game-theory",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Combinatorial Game Theory", "Mathematics", "Strategic Thinking"],
                "description": "Course covering games without chance including combinatorial game theory"
            },
            {
                "title": "Game Theory",
                "provider": "Open Yale Courses",
                "url": "https://oyc.yale.edu/economics/econ-159",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Economics", "Strategic Thinking", "Decision Making"],
                "description": "Course covering game theory including economics and strategic thinking"
            },
            {
                "title": "Competitive Strategy and Game Theory",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/competitive-strategy-and-game-theory/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Competitive Strategy", "Strategy", "Strategic Thinking"],
                "description": "Course covering competitive strategy and game theory including strategic thinking"
            },
            {
                "title": "Economics and Game Theory",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/microeconomics-and-game-theory/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Economics", "Microeconomics", "Strategic Thinking"],
                "description": "Course covering economics and game theory including microeconomics"
            },
            {
                "title": "Introduction to Game Theory",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/game-theory",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Introduction", "Strategic Thinking", "Economics"],
                "description": "Course covering introduction to game theory including strategic thinking"
            },
            {
                "title": "Mathematical Game Theory",
                "provider": "edX",
                "url": "https://www.edx.org/course/mathematical-game-theory",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Game Theory", "Mathematical Game Theory", "Mathematics", "Strategic Thinking"],
                "description": "Course covering mathematical game theory including mathematics and strategic thinking"
            }
        ]
        
        log.info(f"📚 Adding {len(game_theory_courses)} Game Theory courses to knowledge base...")
        return self.batch_add_courses(game_theory_courses)
    
    def add_instructional_design_courses(self) -> bool:
        """
        Add Instructional Design courses to knowledge base.
        Includes 10 courses covering instructional design foundations, e-learning, Articulate Storyline, and needs analysis.
        
        Returns:
            True if successful, False otherwise
        """
        instructional_design_courses = [
            {
                "title": "Instructional Design Foundations and Applications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/instructional-design-foundations-applications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Learning Design", "Education", "Training"],
                "description": "Course covering instructional design foundations and applications including learning design"
            },
            {
                "title": "Learning How to Learn",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/learning-how-to-learn",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Learning", "Learning Strategies", "Education", "Study Skills"],
                "description": "Course covering learning how to learn including learning strategies and study skills"
            },
            {
                "title": "e-Learning Ecologies: Innovative Approaches to Teaching",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/elearning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["E-Learning", "Online Learning", "Teaching", "Education Technology"],
                "description": "Course covering e-learning ecologies including innovative approaches to teaching"
            },
            {
                "title": "Become an Instructional Designer",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/paths/become-an-instructional-designer",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Learning Design", "Career", "Education"],
                "description": "Learning path covering become an instructional designer including career"
            },
            {
                "title": "Instructional Design for Active Learning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/instructional-design-for-active-learning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Active Learning", "Learning Design", "Teaching"],
                "description": "Course covering instructional design for active learning including learning design"
            },
            {
                "title": "Articulate Storyline 360: The Essentials",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/articulate-storyline-360-the-essentials/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Articulate Storyline", "E-Learning", "Instructional Design", "Course Authoring"],
                "description": "Course covering Articulate Storyline 360 including the essentials of course authoring"
            },
            {
                "title": "Instructional Design Master Class",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/instructional-design-master-class/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Learning Design", "Master Class", "Education"],
                "description": "Course covering instructional design master class including learning design"
            },
            {
                "title": "MicroMasters® in Instructional Design and Technology",
                "provider": "edX",
                "url": "https://www.edx.org/micromasters/umgc-instructional-design-and-technology",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Educational Technology", "MicroMasters", "Learning Design"],
                "description": "MicroMasters covering instructional design and technology including educational technology"
            },
            {
                "title": "Instructional Design: Needs Analysis",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/instructional-design-needs-analysis",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Needs Analysis", "Learning Design", "Training"],
                "description": "Course covering instructional design needs analysis including learning design"
            },
            {
                "title": "Professional Certificate in Instructional Design",
                "provider": "Emeritus",
                "url": "https://emeritus.org/university-certificate-programs/instructional-design/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Design", "Professional Certificate", "Learning Design", "Education"],
                "description": "Course covering professional certificate in instructional design including learning design"
            }
        ]
        
        log.info(f"📚 Adding {len(instructional_design_courses)} Instructional Design courses to knowledge base...")
        return self.batch_add_courses(instructional_design_courses)
    
    def add_curriculum_development_courses(self) -> bool:
        """
        Add Curriculum Development courses to knowledge base.
        Includes 10 courses covering curriculum design, outcome-based education, lesson planning, and teaching strategies.
        
        Returns:
            True if successful, False otherwise
        """
        curriculum_courses = [
            {
                "title": "Curriculum Design and Instruction",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teach",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Curriculum Design", "Instruction", "Teaching", "Education"],
                "description": "Course covering curriculum design and instruction including teaching"
            },
            {
                "title": "Uncommon Sense Teaching",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/uncommon-sense-teaching",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Teaching", "Pedagogy", "Education", "Teaching Methods"],
                "description": "Specialization covering uncommon sense teaching including pedagogy"
            },
            {
                "title": "Get Organized: How to be a Together Teacher",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/organized-teacher",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Teaching", "Organization", "Teacher Development", "Education"],
                "description": "Course covering get organized including how to be a together teacher"
            },
            {
                "title": "Outcome-Based Education (OBE) & Academic Quality Assurance",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/outcome-based-education-obe-and-academic-quality-assurance/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Outcome-Based Education", "OBE", "Curriculum Design", "Quality Assurance"],
                "description": "Course covering outcome-based education including OBE and academic quality assurance"
            },
            {
                "title": "Curriculum Development: Instructional Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/curriculum-development/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Curriculum Development", "Instructional Design", "Learning Design", "Education"],
                "description": "Course covering curriculum development including instructional design"
            },
            {
                "title": "Foundations of Teaching for Learning: Curriculum",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teaching-learning-curriculum",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Curriculum", "Teaching", "Education", "Learning"],
                "description": "Course covering foundations of teaching for learning including curriculum"
            },
            {
                "title": "Designing Learning Innovation",
                "provider": "edX",
                "url": "https://www.edx.org/course/designing-learning-innovation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Learning Design", "Innovation", "Curriculum Design", "Education"],
                "description": "Course covering designing learning innovation including curriculum design"
            },
            {
                "title": "Lesson Planning for Teachers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/lesson-planning-for-teachers/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Lesson Planning", "Teaching", "Curriculum Design", "Education"],
                "description": "Course covering lesson planning for teachers including curriculum design"
            },
            {
                "title": "Advanced Instructional Strategies in the Virtual Classroom",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/instructional-strategies-virtual-classroom",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Strategies", "Virtual Classroom", "Online Teaching", "Teaching"],
                "description": "Course covering advanced instructional strategies in the virtual classroom including online teaching"
            },
            {
                "title": "Teaching Online: Reflections on Practice",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teaching-online",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Online Teaching", "Teaching", "E-Learning", "Education"],
                "description": "Course covering teaching online including reflections on practice"
            }
        ]
        
        log.info(f"📚 Adding {len(curriculum_courses)} Curriculum Development courses to knowledge base...")
        return self.batch_add_courses(curriculum_courses)
    
    def add_edtech_tools_courses(self) -> bool:
        """
        Add EdTech Tools & Integration courses to knowledge base.
        Includes 10 courses covering blended learning, Web 2.0 tools, Google Certified Educator, gamification, and Moodle.
        
        Returns:
            True if successful, False otherwise
        """
        edtech_courses = [
            {
                "title": "K-12 Blended & Online Learning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/k-12-online-education",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Blended Learning", "Online Learning", "K-12 Education", "EdTech"],
                "description": "Course covering K-12 blended and online learning including EdTech"
            },
            {
                "title": "Powerful Tools for Teaching and Learning: Web 2.0 Tools",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teaching-learning-tools",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Web 2.0 Tools", "EdTech", "Teaching Tools", "Education Technology"],
                "description": "Course covering powerful tools for teaching and learning including Web 2.0 tools"
            },
            {
                "title": "Google Certified Educator Level 1",
                "provider": "Google for Education",
                "url": "https://edu.google.com/teacher-center/certifications/educator-level1/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Google for Education", "EdTech", "Google Tools", "Education Technology"],
                "description": "Course covering Google Certified Educator Level 1 including Google tools"
            },
            {
                "title": "Gamification",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gamification",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Gamification", "EdTech", "Game-Based Learning", "Education Technology"],
                "description": "Course covering gamification including game-based learning and education technology"
            },
            {
                "title": "Online Education: The Foundations of Online Teaching",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/online-education",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Online Education", "Online Teaching", "E-Learning", "EdTech"],
                "description": "Course covering online education including the foundations of online teaching"
            },
            {
                "title": "Using Educational Technology in the English Language Classroom",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/esl-edtech",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["EdTech", "ESL", "English Language Teaching", "Education Technology"],
                "description": "Course covering using educational technology in the English language classroom including ESL"
            },
            {
                "title": "Introduction to Online and Blended Teaching",
                "provider": "edX",
                "url": "https://www.edx.org/course/introduction-to-online-and-blended-teaching",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Online Teaching", "Blended Learning", "E-Learning", "EdTech"],
                "description": "Course covering introduction to online and blended teaching including EdTech"
            },
            {
                "title": "Moodle 4.0 for Teachers and Administrators",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/moodle-tutorial/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Moodle", "LMS", "Learning Management System", "EdTech"],
                "description": "Course covering Moodle 4.0 for teachers and administrators including LMS"
            },
            {
                "title": "Tech Integration for the Classroom",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/technology-in-the-classroom/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Technology Integration", "EdTech", "Classroom Technology", "Education Technology"],
                "description": "Course covering tech integration for the classroom including education technology"
            },
            {
                "title": "Artificial Intelligence (AI) in Education for Teachers",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/artificial-intelligence-in-education-for-teachers/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AI in Education", "Artificial Intelligence", "EdTech", "Education Technology"],
                "description": "Course covering artificial intelligence in education for teachers including EdTech"
            }
        ]
        
        log.info(f"📚 Adding {len(edtech_courses)} EdTech Tools & Integration courses to knowledge base...")
        return self.batch_add_courses(edtech_courses)
    
    def add_adult_learning_theory_courses(self) -> bool:
        """
        Add Adult Learning Theory (Andragogy) courses to knowledge base.
        Includes 10 courses covering adult learning theory, training the trainer, facilitation skills, and corporate training.
        
        Returns:
            True if successful, False otherwise
        """
        adult_learning_courses = [
            {
                "title": "Foundations of Teaching for Learning: Introduction",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/teaching-learning-introduction",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Teaching", "Learning", "Education", "Pedagogy"],
                "description": "Course covering foundations of teaching for learning including introduction"
            },
            {
                "title": "Adult Learning: Theory and Practice",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adult-learning-theory-and-practice/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Adult Learning", "Andragogy", "Learning Theory", "Training"],
                "description": "Course covering adult learning theory and practice including andragogy"
            },
            {
                "title": "Training the Trainer (T3)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/training-the-trainer-t3/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Train the Trainer", "Training", "Facilitation", "Corporate Training"],
                "description": "Course covering training the trainer including facilitation and corporate training"
            },
            {
                "title": "Instructional Methods in Health Professions Education",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/instructional-methods-health-professions-education",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Instructional Methods", "Health Professions Education", "Teaching", "Medical Education"],
                "description": "Course covering instructional methods in health professions education including medical education"
            },
            {
                "title": "Learning Theories and Instructional Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/learning-theories-and-instructional-design/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Learning Theories", "Instructional Design", "Pedagogy", "Education"],
                "description": "Course covering learning theories and instructional design including pedagogy"
            },
            {
                "title": "Understanding Classroom Interaction",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/classroom-interaction",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Classroom Interaction", "Teaching", "Pedagogy", "Education"],
                "description": "Course covering understanding classroom interaction including pedagogy"
            },
            {
                "title": "Facilitation Skills for Training and Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/facilitation-skills-training/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Facilitation", "Training", "Training and Development", "Corporate Training"],
                "description": "Course covering facilitation skills for training and development including corporate training"
            },
            {
                "title": "Corporate Training: Managing Training & Development",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/corporate-training-managing-training-development/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Training", "Training and Development", "HR", "Learning and Development"],
                "description": "Course covering corporate training including managing training and development"
            },
            {
                "title": "The Science of Learning",
                "provider": "edX",
                "url": "https://www.edx.org/course/the-science-of-learning-what-every-teacher-should-know",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Learning Science", "Pedagogy", "Education", "Teaching"],
                "description": "Course covering the science of learning including what every teacher should know"
            },
            {
                "title": "Design and Development of Educational Technology",
                "provider": "edX",
                "url": "https://www.edx.org/course/design-and-development-of-educational-technology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Educational Technology", "EdTech", "Learning Design", "Education Technology"],
                "description": "Course covering design and development of educational technology including EdTech"
            }
        ]
        
        log.info(f"📚 Adding {len(adult_learning_courses)} Adult Learning Theory (Andragogy) courses to knowledge base...")
        return self.batch_add_courses(adult_learning_courses)
    
    def add_personal_investing_courses(self) -> bool:
        """
        Add Personal Investing (Stocks, ETFs, Real Estate) courses to knowledge base.
        Includes 10 courses covering financial markets, trading, stock investing, real estate investing, and behavioral finance.
        
        Returns:
            True if successful, False otherwise
        """
        personal_investing_courses = [
            {
                "title": "Financial Markets",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/financial-markets-global",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Markets", "Investing", "Finance", "Stock Market"],
                "description": "Course covering financial markets including global financial markets"
            },
            {
                "title": "Practical Guide to Trading: Technical Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/practical-guide-to-trading",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Trading", "Technical Analysis", "Stock Trading", "Investing"],
                "description": "Course covering practical guide to trading including technical analysis"
            },
            {
                "title": "Investing in Stocks: The Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/stock-market-for-beginners-investing-in-stocks/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Stock Investing", "Investing", "Stock Market", "Personal Finance"],
                "description": "Course covering investing in stocks including the complete course for beginners"
            },
            {
                "title": "Real Estate Investing: How to Find Good Deals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/real-estate-investing-how-to-find-good-deals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Real Estate Investing", "Real Estate", "Investing", "Property Investment"],
                "description": "Course covering real estate investing including how to find good deals"
            },
            {
                "title": "Investment Management Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/investment-management-python",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Investment Management", "Portfolio Management", "Investing", "Finance"],
                "description": "Specialization covering investment management including Python for portfolio management"
            },
            {
                "title": "Stock Valuation with Comparable Companies Analysis",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/stock-valuation-comparables",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Stock Valuation", "Financial Analysis", "Investing", "Equity Analysis"],
                "description": "Course covering stock valuation with comparable companies analysis including equity analysis"
            },
            {
                "title": "The Complete Real Estate Developer Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-real-estate-developer-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Real Estate Development", "Real Estate", "Property Development", "Investing"],
                "description": "Course covering the complete real estate developer course including property development"
            },
            {
                "title": "Behavioral Finance",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/behavioral-finance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Behavioral Finance", "Finance", "Investing", "Psychology of Investing"],
                "description": "Course covering behavioral finance including psychology of investing"
            },
            {
                "title": "Introduction to Investments",
                "provider": "edX",
                "url": "https://www.edx.org/course/introduction-to-investments",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Investing", "Introduction", "Personal Finance", "Investment Basics"],
                "description": "Course covering introduction to investments including investment basics"
            },
            {
                "title": "Value Investing: The Complete Value Investing Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/value-investing-bootcamp/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Value Investing", "Investing", "Stock Investing", "Investment Strategy"],
                "description": "Course covering value investing including the complete value investing course"
            }
        ]
        
        log.info(f"📚 Adding {len(personal_investing_courses)} Personal Investing (Stocks, ETFs, Real Estate) courses to knowledge base...")
        return self.batch_add_courses(personal_investing_courses)
    
    def add_financial_planning_budgeting_courses(self) -> bool:
        """
        Add Financial Planning & Budgeting courses to knowledge base.
        Includes 10 courses covering personal finance, budgeting, cash flow management, and financial planning.
        
        Returns:
            True if successful, False otherwise
        """
        financial_planning_courses = [
            {
                "title": "Personal & Family Financial Planning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/family-planning",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Planning", "Personal Finance", "Family Finance", "Budgeting"],
                "description": "Course covering personal and family financial planning including budgeting"
            },
            {
                "title": "Personal Finance Masterclass",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/personal-finance-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Personal Finance", "Financial Planning", "Money Management", "Budgeting"],
                "description": "Course covering personal finance masterclass including money management"
            },
            {
                "title": "Finance for Everyone: Smart Tools for Decision-Making",
                "provider": "edX",
                "url": "https://www.edx.org/course/finance-for-everyone-smart-tools-for-decision-making",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Finance", "Personal Finance", "Financial Decision Making", "Money Management"],
                "description": "Course covering finance for everyone including smart tools for decision-making"
            },
            {
                "title": "Complete Personal Finance Course: Save, Protect, Make More",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/personal-finance-complete-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Personal Finance", "Financial Planning", "Money Management", "Saving"],
                "description": "Course covering complete personal finance course including save, protect, make more"
            },
            {
                "title": "Budgeting and Cash Flow Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/budgeting-and-cash-flow-management/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Budgeting", "Cash Flow Management", "Personal Finance", "Money Management"],
                "description": "Course covering budgeting and cash flow management including money management"
            },
            {
                "title": "Planning for Your Financial Future",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/financial-future",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Planning", "Personal Finance", "Financial Future", "Retirement Planning"],
                "description": "Course covering planning for your financial future including retirement planning"
            },
            {
                "title": "Managing Your Money: MBA Insights for Undergrads",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/managing-money",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Money Management", "Personal Finance", "Financial Planning", "Budgeting"],
                "description": "Course covering managing your money including MBA insights for undergrads"
            },
            {
                "title": "The Complete Financial Analyst Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-financial-analyst-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Analysis", "Finance", "Financial Planning", "Financial Modeling"],
                "description": "Course covering the complete financial analyst course including financial modeling"
            },
            {
                "title": "Financial Literacy",
                "provider": "Alison",
                "url": "https://alison.com/course/financial-literacy",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Literacy", "Personal Finance", "Money Management", "Financial Education"],
                "description": "Course covering financial literacy including personal finance and money management"
            },
            {
                "title": "Personal Finance",
                "provider": "edX",
                "url": "https://www.edx.org/course/personal-finance",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Personal Finance", "Financial Planning", "Money Management", "Budgeting"],
                "description": "Course covering personal finance including financial planning and budgeting"
            }
        ]
        
        log.info(f"📚 Adding {len(financial_planning_courses)} Financial Planning & Budgeting courses to knowledge base...")
        return self.batch_add_courses(financial_planning_courses)
    
    def add_tax_strategy_courses(self) -> bool:
        """
        Add Tax Strategy (Primarily US/International Concepts) courses to knowledge base.
        Includes 10 courses covering US federal taxation, tax preparation, international taxation, and tax strategy.
        
        Returns:
            True if successful, False otherwise
        """
        tax_strategy_courses = [
            {
                "title": "US Federal Taxation Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/us-federal-taxation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Taxation", "US Federal Tax", "Tax Strategy", "Tax Planning"],
                "description": "Specialization covering US federal taxation including tax strategy and planning"
            },
            {
                "title": "Tax Preparation: Learn How to Prepare Taxes",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/tax-preparation-learn-how-to-prepare-taxes/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tax Preparation", "Taxation", "Tax Filing", "Tax Strategy"],
                "description": "Course covering tax preparation including learning how to prepare taxes"
            },
            {
                "title": "Rethinking International Tax Law",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/international-taxation",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["International Taxation", "Tax Law", "Tax Strategy", "Tax Planning"],
                "description": "Course covering rethinking international tax law including tax strategy"
            },
            {
                "title": "Federal Taxation I: Individuals, Employees, and Sole Proprietors",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/federal-taxation-individuals",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Federal Taxation", "Taxation", "Tax Strategy", "Personal Tax"],
                "description": "Course covering federal taxation I including individuals, employees and sole proprietors"
            },
            {
                "title": "Taxes for Small Businesses",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/taxes-for-small-businesses/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Taxes", "Small Business Tax", "Taxation", "Tax Strategy"],
                "description": "Course covering taxes for small businesses including tax strategy"
            },
            {
                "title": "Tax Strategy: How to Pay Less Tax (UK Focus but adaptable)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/tax-strategy/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Tax Strategy", "Tax Planning", "Taxation", "Tax Optimization"],
                "description": "Course covering tax strategy including how to pay less tax"
            },
            {
                "title": "Introduction to Corporate Finance (Includes Tax Shield)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-corporate-finance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Finance", "Tax Shield", "Finance", "Tax Strategy"],
                "description": "Course covering introduction to corporate finance including tax shield"
            },
            {
                "title": "Income Tax Preparation (US)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/income-tax-preparation/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Income Tax", "Tax Preparation", "US Tax", "Taxation"],
                "description": "Course covering income tax preparation including US tax"
            },
            {
                "title": "Understanding Tax in the UK",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/understanding-tax-uk",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["UK Tax", "Taxation", "Tax Strategy", "Tax Planning"],
                "description": "Course covering understanding tax in the UK including tax strategy"
            },
            {
                "title": "Taxation of Business Entities I: Corporations",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/taxation-business-entities-part-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Taxation", "Business Tax", "Taxation", "Tax Strategy"],
                "description": "Course covering taxation of business entities I including corporations"
            }
        ]
        
        log.info(f"📚 Adding {len(tax_strategy_courses)} Tax Strategy (Primarily US/International Concepts) courses to knowledge base...")
        return self.batch_add_courses(tax_strategy_courses)
    
    def add_retirement_planning_courses(self) -> bool:
        """
        Add Retirement Planning courses to knowledge base.
        Includes 10 courses covering retirement planning, pension planning, FIRE (Financial Independence Retire Early), and estate planning.
        
        Returns:
            True if successful, False otherwise
        """
        retirement_planning_courses = [
            {
                "title": "Retirement Planning: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/retirement-planning-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Retirement Planning", "Financial Planning", "Retirement", "Personal Finance"],
                "description": "Course covering retirement planning including the complete guide"
            },
            {
                "title": "Personal Finance: Retirement Planning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/personal-finance-retirement-planning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Retirement Planning", "Personal Finance", "Financial Planning", "Retirement"],
                "description": "Course covering personal finance including retirement planning"
            },
            {
                "title": "Understanding Pensions and Retirement (UK/Global)",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/pensions",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pensions", "Retirement Planning", "Retirement", "Financial Planning"],
                "description": "Course covering understanding pensions and retirement including UK and global perspectives"
            },
            {
                "title": "Retirement Income Planning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/retirement-income-planning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Retirement Planning", "Retirement Income", "Financial Planning", "Retirement"],
                "description": "Course covering retirement income planning including financial planning"
            },
            {
                "title": "Investing for Retirement",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/portfolio-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Retirement Planning", "Investing", "Portfolio Management", "Financial Planning"],
                "description": "Course covering investing for retirement including portfolio management"
            },
            {
                "title": "Managing Retirement Wealth",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/managing-retirement-wealth/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Retirement Planning", "Wealth Management", "Retirement", "Financial Planning"],
                "description": "Course covering managing retirement wealth including financial planning"
            },
            {
                "title": "Financial Planning for Young Adults",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/financial-planning",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Planning", "Retirement Planning", "Personal Finance", "Young Adults"],
                "description": "Course covering financial planning for young adults including retirement planning"
            },
            {
                "title": "Early Retirement: Financial Independence (FIRE)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/financial-independence-retire-early/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FIRE", "Financial Independence", "Early Retirement", "Retirement Planning"],
                "description": "Course covering early retirement including financial independence retire early (FIRE)"
            },
            {
                "title": "Introduction to Financial Planning (Certified Financial Planner - CFP)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/financial-planning-intro",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Financial Planning", "CFP", "Certified Financial Planner", "Retirement Planning"],
                "description": "Course covering introduction to financial planning including CFP certification"
            },
            {
                "title": "Estate Planning for Everyone",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/estate-planning-for-everyone/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Estate Planning", "Retirement Planning", "Financial Planning", "Estate Management"],
                "description": "Course covering estate planning for everyone including retirement planning"
            }
        ]
        
        log.info(f"📚 Adding {len(retirement_planning_courses)} Retirement Planning courses to knowledge base...")
        return self.batch_add_courses(retirement_planning_courses)
    
    def add_sustainable_business_courses(self) -> bool:
        """
        Add Sustainable Business Strategies courses to knowledge base.
        Includes 10 courses covering sustainable business strategy, circular economy, corporate sustainability, and business transformation.
        
        Returns:
            True if successful, False otherwise
        """
        sustainable_business_courses = [
            {
                "title": "Sustainable Business Strategy",
                "provider": "Harvard Business School Online",
                "url": "https://online.hbs.edu/courses/sustainable-business-strategy/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Business", "Business Strategy", "Sustainability", "ESG"],
                "description": "Course covering sustainable business strategy including ESG"
            },
            {
                "title": "Business Strategies for A Better World",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/business-strategies-better-world",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Business", "Business Strategy", "Social Impact", "Sustainability"],
                "description": "Specialization covering business strategies for a better world including social impact"
            },
            {
                "title": "Strategy and Sustainability",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/strategy-sustainability",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategy", "Sustainability", "Sustainable Business", "Business Strategy"],
                "description": "Course covering strategy and sustainability including sustainable business"
            },
            {
                "title": "Become a Sustainable Business Change Agent",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/become-a-sustainable-business-change-agent",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Business", "Change Management", "Sustainability", "Business Transformation"],
                "description": "Specialization covering become a sustainable business change agent including business transformation"
            },
            {
                "title": "Sustainable Business: Big Issues, Big Changes",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/sustainable-business",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Business", "Sustainability", "Business Strategy", "ESG"],
                "description": "Course covering sustainable business including big issues and big changes"
            },
            {
                "title": "Managing the Company of the Future",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/company-future-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Management", "Sustainable Business", "Future of Business", "Management"],
                "description": "Course covering managing the company of the future including sustainable business"
            },
            {
                "title": "Circular Economy - Sustainable Materials Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/circular-economy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Circular Economy", "Sustainability", "Sustainable Materials", "Waste Management"],
                "description": "Course covering circular economy including sustainable materials management"
            },
            {
                "title": "Corporate Sustainability: Understanding and Seizing the Strategic Opportunity",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/corporate-sustainability",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Sustainability", "Sustainability", "Business Strategy", "ESG"],
                "description": "Course covering corporate sustainability including understanding and seizing the strategic opportunity"
            },
            {
                "title": "Leading Sustainable Business Transformation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/leading-sustainable-business-transformation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Business", "Business Transformation", "Leadership", "Sustainability"],
                "description": "Course covering leading sustainable business transformation including leadership"
            },
            {
                "title": "Business Sustainability Management",
                "provider": "Cambridge",
                "url": "https://www.cisl.cam.ac.uk/education/learn-online/business-sustainability-management-online-course",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Business Sustainability", "Sustainability Management", "Sustainable Business", "ESG"],
                "description": "Course covering business sustainability management including ESG"
            }
        ]
        
        log.info(f"📚 Adding {len(sustainable_business_courses)} Sustainable Business Strategies courses to knowledge base...")
        return self.batch_add_courses(sustainable_business_courses)
    
    def add_climate_change_policy_courses(self) -> bool:
        """
        Add Climate Change & Policy courses to knowledge base.
        Includes 10 courses covering climate change science, sustainable development, climate adaptation, and climate policy.
        
        Returns:
            True if successful, False otherwise
        """
        climate_change_courses = [
            {
                "title": "Global Warming I: The Science and Modeling of Climate Change",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/global-warming",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Change", "Global Warming", "Climate Science", "Environmental Science"],
                "description": "Course covering global warming I including the science and modeling of climate change"
            },
            {
                "title": "Climate Change and Sustainable Investing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/climate-change-and-sustainable-investing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Change", "Sustainable Investing", "ESG Investing", "Climate Finance"],
                "description": "Course covering climate change and sustainable investing including ESG investing"
            },
            {
                "title": "The Age of Sustainable Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/sustainable-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Development", "Climate Change", "Sustainability", "Development"],
                "description": "Course covering the age of sustainable development including climate change"
            },
            {
                "title": "Climate Change: The Science and Global Impact",
                "provider": "edX",
                "url": "https://www.edx.org/course/climate-change-the-science-and-global-impact",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Change", "Climate Science", "Environmental Science", "Global Impact"],
                "description": "Course covering climate change including the science and global impact"
            },
            {
                "title": "Introduction to Sustainability",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/sustainability",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainability", "Climate Change", "Environmental Science", "Sustainable Development"],
                "description": "Course covering introduction to sustainability including climate change"
            },
            {
                "title": "Climate Adaptation in Africa",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/climate-adaptation-africa",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Adaptation", "Climate Change", "Environmental Policy", "Sustainability"],
                "description": "Course covering climate adaptation in Africa including climate change"
            },
            {
                "title": "Our Common Future",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/our-common-future",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainability", "Climate Change", "Sustainable Development", "Environmental Policy"],
                "description": "Course covering our common future including sustainability and climate change"
            },
            {
                "title": "Act on Climate: Steps to Individual, Community, and Political Action",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/act-on-climate",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Action", "Climate Change", "Environmental Policy", "Sustainability"],
                "description": "Course covering act on climate including steps to individual, community and political action"
            },
            {
                "title": "Climate Change Policy and Public Health",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/climate-change-policy-public-health",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Change Policy", "Climate Change", "Public Health", "Environmental Policy"],
                "description": "Course covering climate change policy and public health including environmental policy"
            },
            {
                "title": "Unlocking Investment and Finance in Emerging Markets (Climate)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/unlocking-investment-finance-emerging-markets",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Climate Finance", "Climate Change", "Sustainable Investing", "Emerging Markets"],
                "description": "Course covering unlocking investment and finance in emerging markets including climate"
            }
        ]
        
        log.info(f"📚 Adding {len(climate_change_courses)} Climate Change & Policy courses to knowledge base...")
        return self.batch_add_courses(climate_change_courses)
    
    def add_renewable_energy_courses(self) -> bool:
        """
        Add Renewable Energy Technologies courses to knowledge base.
        Includes 10 courses covering solar energy, wind energy, renewable energy systems, and green building.
        
        Returns:
            True if successful, False otherwise
        """
        renewable_energy_courses = [
            {
                "title": "Renewable Energy and Green Building Entrepreneurship",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/renewable-energy-entrepreneurship",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Renewable Energy", "Green Building", "Sustainability", "Energy Entrepreneurship"],
                "description": "Course covering renewable energy and green building entrepreneurship including energy entrepreneurship"
            },
            {
                "title": "Solar Energy Basics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/solar-energy-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "Renewable Energy", "Energy", "Sustainability"],
                "description": "Course covering solar energy basics including renewable energy"
            },
            {
                "title": "Wind Energy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wind-energy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Wind Energy", "Renewable Energy", "Energy", "Sustainability"],
                "description": "Course covering wind energy including renewable energy"
            },
            {
                "title": "Introduction to Renewable Energy Systems",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/introduction-to-renewable-energy-systems/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Renewable Energy", "Energy Systems", "Sustainability", "Clean Energy"],
                "description": "Course covering introduction to renewable energy systems including clean energy"
            },
            {
                "title": "Solar Energy Engineering MicroMasters",
                "provider": "edX",
                "url": "https://www.edx.org/micromasters/tudelftx-solar-energy-engineering",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "Energy Engineering", "Renewable Energy", "Engineering"],
                "description": "MicroMasters covering solar energy engineering including energy engineering"
            },
            {
                "title": "Energy: The Enterprise",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/energy-enterprise",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Energy", "Renewable Energy", "Energy Business", "Sustainability"],
                "description": "Course covering energy including the enterprise and energy business"
            },
            {
                "title": "Off-Grid Solar Energy Systems",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/off-grid-solar-energy-systems/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "Off-Grid Systems", "Renewable Energy", "Energy Systems"],
                "description": "Course covering off-grid solar energy systems including energy systems"
            },
            {
                "title": "Photovoltaic Solar Energy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/photovoltaic-solar-energy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Solar Energy", "Photovoltaic", "Renewable Energy", "Energy Engineering"],
                "description": "Course covering photovoltaic solar energy including energy engineering"
            },
            {
                "title": "Hydro, Wind & Solar Power: Renewable Energy Systems",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/renewable-energy-systems/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Renewable Energy", "Hydro Power", "Wind Power", "Solar Power"],
                "description": "Course covering hydro, wind and solar power including renewable energy systems"
            },
            {
                "title": "Electric Power Systems",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/electric-power-systems",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Electric Power", "Power Systems", "Energy", "Renewable Energy"],
                "description": "Course covering electric power systems including renewable energy"
            }
        ]
        
        log.info(f"📚 Adding {len(renewable_energy_courses)} Renewable Energy Technologies courses to knowledge base...")
        return self.batch_add_courses(renewable_energy_courses)
    
    def add_csr_courses(self) -> bool:
        """
        Add Corporate Social Responsibility (CSR) courses to knowledge base.
        Includes 10 courses covering CSR, ESG, impact investing, social entrepreneurship, and responsible marketing.
        
        Returns:
            True if successful, False otherwise
        """
        csr_courses = [
            {
                "title": "Corporate Social Responsibility (CSR): A Strategic Approach",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/corporate-social-responsibility",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CSR", "Corporate Social Responsibility", "ESG", "Sustainability"],
                "description": "Course covering corporate social responsibility including a strategic approach and ESG"
            },
            {
                "title": "CSR & ESG: A Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/corporate-social-responsibility-csr/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CSR", "ESG", "Corporate Social Responsibility", "Sustainability"],
                "description": "Course covering CSR and ESG including the complete guide"
            },
            {
                "title": "Sustainable Business: Corporate Social Responsibility",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sustainable-business-corporate-social-responsibility/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CSR", "Sustainable Business", "Corporate Social Responsibility", "Sustainability"],
                "description": "Course covering sustainable business including corporate social responsibility"
            },
            {
                "title": "Impact Investing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/impact-investing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Impact Investing", "Social Impact", "Sustainable Investing", "ESG"],
                "description": "Course covering impact investing including social impact and ESG"
            },
            {
                "title": "Social Entrepreneurship",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-social-entrepreneurship",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Social Entrepreneurship", "Social Impact", "Entrepreneurship", "CSR"],
                "description": "Course covering social entrepreneurship including social impact"
            },
            {
                "title": "Strategy and Sustainability",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/strategy-sustainability",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategy", "Sustainability", "CSR", "Business Strategy"],
                "description": "Course covering strategy and sustainability including CSR"
            },
            {
                "title": "Responsible Marketing and the Future of Business",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/responsible-marketing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Responsible Marketing", "CSR", "Marketing Ethics", "Sustainable Marketing"],
                "description": "Course covering responsible marketing and the future of business including CSR"
            },
            {
                "title": "Greening the Economy: Sustainable Cities",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gte-sustainable-cities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Cities", "Sustainability", "Urban Planning", "CSR"],
                "description": "Course covering greening the economy including sustainable cities"
            },
            {
                "title": "CSR: The Future of Business",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/csr-the-future-of-business/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CSR", "Corporate Social Responsibility", "Future of Business", "Sustainability"],
                "description": "Course covering CSR including the future of business"
            },
            {
                "title": "ESG and Social Activism",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-esg-social-activism",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["ESG", "Social Activism", "CSR", "Corporate Social Responsibility"],
                "description": "Course covering ESG and social activism including CSR"
            }
        ]
        
        log.info(f"📚 Adding {len(csr_courses)} Corporate Social Responsibility (CSR) courses to knowledge base...")
        return self.batch_add_courses(csr_courses)
    
    def add_supply_chain_management_courses(self) -> bool:
        """
        Add Supply Chain Management (SCM) courses to knowledge base.
        Includes 10 courses covering supply chain fundamentals, logistics, analytics, and global supply chain management.
        
        Returns:
            True if successful, False otherwise
        """
        scm_courses = [
            {
                "title": "Supply Chain Management Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/supply-chain-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Management", "SCM", "Logistics", "Operations"],
                "description": "Specialization covering supply chain management including logistics and operations"
            },
            {
                "title": "MicroMasters® Program in Supply Chain Management",
                "provider": "edX",
                "url": "https://www.edx.org/micromasters/mitx-supply-chain-management",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Management", "SCM", "Logistics", "Operations Management"],
                "description": "MicroMasters covering supply chain management including logistics and operations management"
            },
            {
                "title": "Global Supply Chain Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/global-supply-chain-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Supply Chain", "Supply Chain Management", "International Logistics", "SCM"],
                "description": "Specialization covering global supply chain management including international logistics"
            },
            {
                "title": "Supply Chain Analytics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Analytics", "SCM", "Data Analytics", "Supply Chain"],
                "description": "Course covering supply chain analytics including data analytics"
            },
            {
                "title": "Supply Chain Management A-Z",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/supply-chain-management-a-z/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Management", "SCM", "Logistics", "Operations"],
                "description": "Course covering supply chain management A-Z including logistics and operations"
            },
            {
                "title": "Operations and Supply Chain Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/operations-and-supply-chain-management/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operations Management", "Supply Chain Management", "SCM", "Operations"],
                "description": "Course covering operations and supply chain management"
            },
            {
                "title": "Supply Chain Principles",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-principles",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain", "SCM", "Supply Chain Principles", "Logistics"],
                "description": "Course covering supply chain principles including logistics"
            },
            {
                "title": "Supply Chain Logistics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-logistics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Logistics", "Logistics", "SCM", "Supply Chain"],
                "description": "Course covering supply chain logistics including logistics"
            },
            {
                "title": "Introduction to Supply Chain Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-management",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Management", "SCM", "Introduction", "Logistics"],
                "description": "Course covering introduction to supply chain management including logistics"
            }
        ]
        
        log.info(f"📚 Adding {len(scm_courses)} Supply Chain Management (SCM) courses to knowledge base...")
        return self.batch_add_courses(scm_courses)
    
    def add_operations_management_courses(self) -> bool:
        """
        Add Operations Management (Six Sigma, Lean) courses to knowledge base.
        Includes 10 courses covering operations management, Six Sigma, Lean manufacturing, and process improvement.
        
        Returns:
            True if successful, False otherwise
        """
        operations_courses = [
            {
                "title": "Operations Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-operations",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operations Management", "Operations", "Business Operations", "Process Management"],
                "description": "Course covering operations management including business operations"
            },
            {
                "title": "Six Sigma Yellow Belt",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/six-sigma-yellow-belt",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Six Sigma", "Yellow Belt", "Quality Management", "Process Improvement"],
                "description": "Course covering Six Sigma Yellow Belt including quality management"
            },
            {
                "title": "Six Sigma Green Belt Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/six-sigma-green-belt",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Six Sigma", "Green Belt", "Quality Management", "Process Improvement"],
                "description": "Specialization covering Six Sigma Green Belt including quality management"
            },
            {
                "title": "Lean Six Sigma Black Belt",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/lean-six-sigma-black-belt-training-certification/",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Lean Six Sigma", "Black Belt", "Six Sigma", "Quality Management"],
                "description": "Course covering Lean Six Sigma Black Belt including quality management"
            },
            {
                "title": "Operations Analytics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-operations-analytics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operations Analytics", "Operations Management", "Data Analytics", "Business Analytics"],
                "description": "Course covering operations analytics including data analytics"
            },
            {
                "title": "Lean Management & Lean Manufacturing",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/lean-management-lean-manufacturing/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Lean Management", "Lean Manufacturing", "Process Improvement", "Operations"],
                "description": "Course covering lean management and lean manufacturing including process improvement"
            },
            {
                "title": "Operations Management A-Z",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/operations-management-a-z/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operations Management", "Operations", "Business Operations", "Process Management"],
                "description": "Course covering operations management A-Z including business operations"
            },
            {
                "title": "Supply Chain Operations",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-operations",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Operations", "Operations Management", "SCM", "Operations"],
                "description": "Course covering supply chain operations including operations management"
            },
            {
                "title": "Process Improvement",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/process-improvement",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Process Improvement", "Operations Management", "Quality Management", "Six Sigma"],
                "description": "Course covering process improvement including quality management"
            },
            {
                "title": "Introduction to Operations Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/wharton-operations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Operations Management", "Introduction", "Business Operations", "Operations"],
                "description": "Course covering introduction to operations management including business operations"
            }
        ]
        
        log.info(f"📚 Adding {len(operations_courses)} Operations Management (Six Sigma, Lean) courses to knowledge base...")
        return self.batch_add_courses(operations_courses)
    
    def add_procurement_sourcing_courses(self) -> bool:
        """
        Add Procurement & Sourcing courses to knowledge base.
        Includes 10 courses covering procurement, strategic sourcing, negotiation, and global procurement.
        
        Returns:
            True if successful, False otherwise
        """
        procurement_courses = [
            {
                "title": "Procurement and Sourcing Introduction",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/procurement-sourcing",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Procurement", "Sourcing", "Supply Chain", "Purchasing"],
                "description": "Course covering procurement and sourcing introduction including purchasing"
            },
            {
                "title": "Strategic Procurement and Sourcing",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/strategic-procurement-and-sourcing/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategic Procurement", "Sourcing", "Procurement", "Supply Chain"],
                "description": "Course covering strategic procurement and sourcing including supply chain"
            },
            {
                "title": "Supply Chain Sourcing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-sourcing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Sourcing", "Sourcing", "Procurement", "SCM"],
                "description": "Course covering supply chain sourcing including procurement"
            },
            {
                "title": "Procurement Basics: The Good, The Bad, and The Ugly",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/procurement-basics/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Procurement", "Procurement Basics", "Purchasing", "Supply Chain"],
                "description": "Course covering procurement basics including purchasing"
            },
            {
                "title": "Negotiation for Procurement Professionals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/negotiation-for-procurement-professionals/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Procurement Negotiation", "Negotiation", "Procurement", "Sourcing"],
                "description": "Course covering negotiation for procurement professionals including sourcing"
            },
            {
                "title": "Global Procurement and Sourcing Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/global-procurement-sourcing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Global Procurement", "Sourcing", "International Procurement", "Supply Chain"],
                "description": "Specialization covering global procurement and sourcing including international procurement"
            },
            {
                "title": "Procurement Negotiation Strategy",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/procurement-negotiation-strategy/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Procurement Negotiation", "Negotiation Strategy", "Procurement", "Sourcing"],
                "description": "Course covering procurement negotiation strategy including sourcing"
            },
            {
                "title": "Strategic Sourcing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/strategic-sourcing",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Strategic Sourcing", "Sourcing", "Procurement", "Supply Chain"],
                "description": "Course covering strategic sourcing including procurement"
            },
            {
                "title": "Procurement Management",
                "provider": "edX",
                "url": "https://www.edx.org/course/supply-chain-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Procurement Management", "Procurement", "Supply Chain Management", "SCM"],
                "description": "Course covering procurement management including supply chain management"
            },
            {
                "title": "Purchasing and Procurement",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/purchasing-and-procurement/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Purchasing", "Procurement", "Supply Chain", "Sourcing"],
                "description": "Course covering purchasing and procurement including sourcing"
            }
        ]
        
        log.info(f"📚 Adding {len(procurement_courses)} Procurement & Sourcing courses to knowledge base...")
        return self.batch_add_courses(procurement_courses)
    
    def add_warehouse_inventory_management_courses(self) -> bool:
        """
        Add Warehouse & Inventory Management courses to knowledge base.
        Includes 10 courses covering inventory management, warehouse management, logistics, and supply chain planning.
        
        Returns:
            True if successful, False otherwise
        """
        warehouse_courses = [
            {
                "title": "Inventory Management A-Z",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/inventory-management-a-z/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Inventory Management", "Inventory Control", "Warehouse Management", "Operations"],
                "description": "Course covering inventory management A-Z including inventory control"
            },
            {
                "title": "Warehouse Management Fundamentals",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/warehouse-management-fundamentals/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Warehouse Management", "Logistics", "Inventory Management", "Operations"],
                "description": "Course covering warehouse management fundamentals including logistics"
            },
            {
                "title": "Inventory Management & Inventory Control",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/inventory-management-inventory-control/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Inventory Management", "Inventory Control", "Operations", "Supply Chain"],
                "description": "Course covering inventory management and inventory control including supply chain"
            },
            {
                "title": "Supply Chain Logistics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-logistics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Logistics", "Logistics", "Warehouse Management", "SCM"],
                "description": "Course covering supply chain logistics including warehouse management"
            },
            {
                "title": "Operations Management: Inventory Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/operations-management-inventory-management/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Inventory Management", "Operations Management", "Operations", "Supply Chain"],
                "description": "Course covering operations management including inventory management"
            },
            {
                "title": "Warehouse Management: Inventory, Stock & Supply Chain",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/warehouse-management-inventory-stock-supply-chain/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Warehouse Management", "Inventory Management", "Supply Chain", "Logistics"],
                "description": "Course covering warehouse management including inventory, stock and supply chain"
            },
            {
                "title": "Basics of Logistics, Supply Chain & Warehouse Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/basics-of-logistics-supply-chain-warehouse-management/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Logistics", "Supply Chain", "Warehouse Management", "SCM"],
                "description": "Course covering basics of logistics, supply chain and warehouse management"
            },
            {
                "title": "Supply Chain Planning",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/supply-chain-planning",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Supply Chain Planning", "Inventory Management", "SCM", "Operations"],
                "description": "Course covering supply chain planning including inventory management"
            },
            {
                "title": "Logistics and Supply Chain Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/logistics-supply-chain-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Logistics", "Supply Chain Management", "Warehouse Management", "SCM"],
                "description": "Course covering logistics and supply chain management including warehouse management"
            },
            {
                "title": "Inventory Strategy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/inventory-strategy",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Inventory Strategy", "Inventory Management", "Supply Chain", "Operations"],
                "description": "Course covering inventory strategy including inventory management"
            }
        ]
        
        log.info(f"📚 Adding {len(warehouse_courses)} Warehouse & Inventory Management courses to knowledge base...")
        return self.batch_add_courses(warehouse_courses)
    
    def add_civil_engineering_courses(self) -> bool:
        """
        Add Civil Engineering (Structural & Transportation) courses to knowledge base.
        Includes 10 courses covering structural analysis, mechanics of materials, highway planning, and reinforced concrete design.
        
        Returns:
            True if successful, False otherwise
        """
        civil_engineering_courses = [
            {
                "title": "Introduction to Engineering Mechanics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/engineering-mechanics-statics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Mechanics", "Statics", "Civil Engineering", "Structural Engineering"],
                "description": "Course covering introduction to engineering mechanics including statics"
            },
            {
                "title": "Mechanics of Materials I: Fundamentals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/mechanics-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mechanics of Materials", "Structural Engineering", "Civil Engineering", "Materials Science"],
                "description": "Course covering mechanics of materials I including fundamentals"
            },
            {
                "title": "Fundamentals of Structural Analysis",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/fundamentals-of-structural-analysis/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Structural Analysis", "Civil Engineering", "Structural Engineering", "Engineering"],
                "description": "Course covering fundamentals of structural analysis including structural engineering"
            },
            {
                "title": "Civil Engineering: Structural Analysis",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/civil-engineering-structural-analysis/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Civil Engineering", "Structural Analysis", "Structural Engineering", "Engineering"],
                "description": "Course covering civil engineering including structural analysis"
            },
            {
                "title": "Highway Planning, Pavement Design and Construction",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/highway-planning-pavement-design-construction",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Highway Planning", "Transportation Engineering", "Civil Engineering", "Pavement Design"],
                "description": "Specialization covering highway planning, pavement design and construction including transportation engineering"
            },
            {
                "title": "Materials Science: 10 Things Every Engineer Should Know",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/materials-science",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Materials Science", "Engineering", "Civil Engineering", "Materials"],
                "description": "Course covering materials science including 10 things every engineer should know"
            },
            {
                "title": "Introduction to Air Quality (Environmental Civil)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intro-to-air-quality",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Air Quality", "Environmental Engineering", "Civil Engineering", "Environmental Science"],
                "description": "Course covering introduction to air quality including environmental civil engineering"
            },
            {
                "title": "Mastering Statics: Engineering Mechanics",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mastering-statics/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Statics", "Engineering Mechanics", "Civil Engineering", "Structural Engineering"],
                "description": "Course covering mastering statics including engineering mechanics"
            },
            {
                "title": "Reinforced Concrete Design",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/reinforced-concrete-design/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Reinforced Concrete", "Structural Design", "Civil Engineering", "Concrete Design"],
                "description": "Course covering reinforced concrete design including structural design"
            },
            {
                "title": "Roadway Design with InfraWorks 360",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/roadway-design-with-infraworks-360",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Roadway Design", "Transportation Engineering", "InfraWorks", "Civil Engineering"],
                "description": "Course covering roadway design with InfraWorks 360 including transportation engineering"
            }
        ]
        
        log.info(f"📚 Adding {len(civil_engineering_courses)} Civil Engineering (Structural & Transportation) courses to knowledge base...")
        return self.batch_add_courses(civil_engineering_courses)
    
    def add_construction_management_courses(self) -> bool:
        """
        Add Construction Management courses to knowledge base.
        Includes 10 courses covering construction project management, scheduling, cost estimating, and construction finance.
        
        Returns:
            True if successful, False otherwise
        """
        construction_management_courses = [
            {
                "title": "Construction Management Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/construction-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Management", "Project Management", "Construction", "Management"],
                "description": "Specialization covering construction management including project management"
            },
            {
                "title": "Construction Project Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/construction-project-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Project Management", "Project Management", "Construction", "Management"],
                "description": "Course covering construction project management including project management"
            },
            {
                "title": "Construction Scheduling",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/construction-scheduling",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Scheduling", "Project Scheduling", "Construction Management", "Planning"],
                "description": "Course covering construction scheduling including project scheduling"
            },
            {
                "title": "Construction Cost Estimating and Cost Control",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/construction-cost-estimating",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Cost Estimating", "Cost Control", "Construction Management", "Project Management"],
                "description": "Course covering construction cost estimating and cost control including project management"
            },
            {
                "title": "Construction Finance",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/construction-finance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Finance", "Finance", "Construction Management", "Financial Management"],
                "description": "Course covering construction finance including financial management"
            },
            {
                "title": "Construction Management Foundations",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/construction-management-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Management", "Foundations", "Construction", "Management"],
                "description": "Course covering construction management foundations including management"
            },
            {
                "title": "Major Engineering Project Performance",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/major-engineering-project-performance",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Engineering Project Management", "Project Performance", "Construction Management", "Project Management"],
                "description": "Course covering major engineering project performance including project management"
            },
            {
                "title": "Safety in the Construction Industry",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/construction-safety",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Safety", "Safety Management", "Construction Management", "OSHA"],
                "description": "Course covering safety in the construction industry including safety management"
            },
            {
                "title": "Construction Estimating and Costing",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/construction-estimating-and-costing/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Construction Estimating", "Costing", "Construction Management", "Cost Estimation"],
                "description": "Course covering construction estimating and costing including cost estimation"
            },
            {
                "title": "Project Management for Construction",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/project-management-for-construction/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Construction", "Construction Management", "Project Planning"],
                "description": "Course covering project management for construction including project planning"
            }
        ]
        
        log.info(f"📚 Adding {len(construction_management_courses)} Construction Management courses to knowledge base...")
        return self.batch_add_courses(construction_management_courses)
    
    def add_architecture_urban_planning_courses(self) -> bool:
        """
        Add Architecture & Urban Planning courses to knowledge base.
        Includes 10 courses covering architecture, urban design, sustainable cities, and smart cities.
        
        Returns:
            True if successful, False otherwise
        """
        architecture_courses = [
            {
                "title": "The Architectural Imagination",
                "provider": "edX",
                "url": "https://www.edx.org/course/the-architectural-imagination",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Architecture", "Architectural Design", "Design", "Urban Planning"],
                "description": "Course covering the architectural imagination including architectural design"
            },
            {
                "title": "Making Architecture",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/making-architecture",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Architecture", "Architectural Design", "Design", "Architecture Basics"],
                "description": "Course covering making architecture including architectural design"
            },
            {
                "title": "Roman Architecture",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/roman-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Architecture", "Roman Architecture", "Architectural History", "Design"],
                "description": "Course covering Roman architecture including architectural history"
            },
            {
                "title": "Greening the Economy: Sustainable Cities",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gte-sustainable-cities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sustainable Cities", "Urban Planning", "Sustainability", "City Planning"],
                "description": "Course covering greening the economy including sustainable cities"
            },
            {
                "title": "Management of Urban Infrastructures",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/urban-infrastructures",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Urban Infrastructure", "Urban Planning", "Infrastructure Management", "City Planning"],
                "description": "Course covering management of urban infrastructures including city planning"
            },
            {
                "title": "Smart Cities",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/smart-cities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Smart Cities", "Urban Planning", "City Planning", "Technology"],
                "description": "Course covering smart cities including urban planning and technology"
            },
            {
                "title": "Urban Design for the Public Good",
                "provider": "edX",
                "url": "https://www.edx.org/course/urban-design-for-the-public-good",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Urban Design", "Urban Planning", "City Planning", "Public Design"],
                "description": "Course covering urban design for the public good including city planning"
            },
            {
                "title": "Eco-Design for Cities and Suburbs",
                "provider": "edX",
                "url": "https://www.edx.org/course/eco-design-for-cities-and-suburbs",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Eco-Design", "Urban Planning", "Sustainable Design", "City Planning"],
                "description": "Course covering eco-design for cities and suburbs including sustainable design"
            },
            {
                "title": "Frank Gehry Teaches Design and Architecture",
                "provider": "MasterClass",
                "url": "https://www.masterclass.com/classes/frank-gehry-teaches-design-and-architecture",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Architecture", "Architectural Design", "Design", "Creative Design"],
                "description": "Course covering Frank Gehry teaches design and architecture including creative design"
            },
            {
                "title": "Designing Cities",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/designing-cities",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Urban Planning", "City Design", "City Planning", "Urban Design"],
                "description": "Course covering designing cities including urban planning and city design"
            }
        ]
        
        log.info(f"📚 Adding {len(architecture_courses)} Architecture & Urban Planning courses to knowledge base...")
        return self.batch_add_courses(architecture_courses)
    
    def add_bim_courses(self) -> bool:
        """
        Add BIM (Building Information Modeling) courses to knowledge base.
        Includes 10 courses covering BIM fundamentals, Revit, AutoCAD, Navisworks, and 3D CAD.
        
        Returns:
            True if successful, False otherwise
        """
        bim_courses = [
            {
                "title": "BIM Fundamentals for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bim-fundamentals-for-engineers",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["BIM", "Building Information Modeling", "Revit", "Architecture"],
                "description": "Course covering BIM fundamentals for engineers including building information modeling"
            },
            {
                "title": "Autodesk Certified Professional: Revit for Architectural Design",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/autodesk-revit-architectural-design-exam-prep",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Revit", "BIM", "Autodesk", "Architectural Design"],
                "description": "Course covering Autodesk certified professional Revit for architectural design including BIM"
            },
            {
                "title": "BIM Application for Engineers",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bim-application-for-engineers",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["BIM", "Building Information Modeling", "Engineering", "Revit"],
                "description": "Course covering BIM application for engineers including building information modeling"
            },
            {
                "title": "The Complete AutoCAD 2018-2024 Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/autocad-2018-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AutoCAD", "CAD", "Drafting", "Engineering"],
                "description": "Course covering the complete AutoCAD 2018-2024 course including CAD and drafting"
            },
            {
                "title": "Revit Architecture: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/revit-architecture-the-complete-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Revit", "BIM", "Architecture", "Architectural Design"],
                "description": "Course covering Revit architecture including the complete guide and BIM"
            },
            {
                "title": "BIM 4D and 5D with Navisworks",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/bim-4d-and-5d-with-navisworks/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["BIM", "Navisworks", "4D BIM", "5D BIM", "Building Information Modeling"],
                "description": "Course covering BIM 4D and 5D with Navisworks including building information modeling"
            },
            {
                "title": "Learning Revit 2024",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/learning-revit-2024",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Revit", "BIM", "Architecture", "Building Information Modeling"],
                "description": "Course covering learning Revit 2024 including BIM and building information modeling"
            },
            {
                "title": "3D CAD Fundamental",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/3d-cad-fundamental",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["3D CAD", "CAD", "3D Modeling", "Engineering"],
                "description": "Course covering 3D CAD fundamental including 3D modeling"
            },
            {
                "title": "AutoCAD: Tips & Tricks",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/autocad-tips-tricks",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["AutoCAD", "CAD", "Drafting", "Engineering"],
                "description": "Course covering AutoCAD tips and tricks including CAD and drafting"
            },
            {
                "title": "Revit Structure: From Zero to Hero",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/revit-structure-from-zero-to-hero/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Revit Structure", "BIM", "Structural Engineering", "Revit"],
                "description": "Course covering Revit structure including from zero to hero and BIM"
            }
        ]
        
        log.info(f"📚 Adding {len(bim_courses)} BIM (Building Information Modeling) courses to knowledge base...")
        return self.batch_add_courses(bim_courses)
    
    def add_bioinformatics_courses(self) -> bool:
        """
        Add Bioinformatics courses to knowledge base.
        Includes 10 courses covering bioinformatics, genomic data science, Python for genomics, and RNA-seq.
        
        Returns:
            True if successful, False otherwise
        """
        bioinformatics_courses = [
            {
                "title": "Bioinformatics Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/bioinformatics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "Genomics", "Data Science", "Biology"],
                "description": "Specialization covering bioinformatics including genomics and data science"
            },
            {
                "title": "Genomic Data Science Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/genomic-data-science",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Genomic Data Science", "Bioinformatics", "Genomics", "Data Science"],
                "description": "Specialization covering genomic data science including bioinformatics"
            },
            {
                "title": "Biology Meets Programming: Bioinformatics for Beginners",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bioinformatics-pku",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "Programming", "Biology", "Python"],
                "description": "Course covering biology meets programming including bioinformatics for beginners"
            },
            {
                "title": "Python for Genomic Data Science",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/python-genomics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Python", "Genomic Data Science", "Bioinformatics", "Genomics"],
                "description": "Course covering Python for genomic data science including bioinformatics"
            },
            {
                "title": "Bioinformatics: Introduction and Methods",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bioinformatics-pku",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "Introduction", "Genomics", "Biology"],
                "description": "Course covering bioinformatics including introduction and methods"
            },
            {
                "title": "Introduction to Bioinformatics and RNA-seq",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/introduction-to-bioinformatics-and-rna-seq/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "RNA-seq", "Genomics", "Data Analysis"],
                "description": "Course covering introduction to bioinformatics and RNA-seq including genomics"
            },
            {
                "title": "Plant Bioinformatics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/plant-bioinformatics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Plant Bioinformatics", "Bioinformatics", "Plant Biology", "Genomics"],
                "description": "Course covering plant bioinformatics including plant biology and genomics"
            },
            {
                "title": "Industrial Biotechnology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/industrial-biotech",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Industrial Biotechnology", "Biotechnology", "Bioinformatics", "Biology"],
                "description": "Course covering industrial biotechnology including biotechnology"
            },
            {
                "title": "Bioinformatics Capstone: Big Data in Biology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bioinformatics-project",
                "difficulty": "Advanced",

                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "Big Data", "Genomics", "Data Science"],
                "description": "Course covering bioinformatics capstone including big data in biology"
            },
            {
                "title": "Learn Bioinformatics from Scratch (Theory & Python)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/learn-bioinformatics-from-scratch/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bioinformatics", "Python", "Genomics", "Biology"],
                "description": "Course covering learn bioinformatics from scratch including theory and Python"
            }
        ]
        
        log.info(f"📚 Adding {len(bioinformatics_courses)} Bioinformatics courses to knowledge base...")
        return self.batch_add_courses(bioinformatics_courses)
    
    def add_biotechnology_fundamentals_courses(self) -> bool:
        """
        Add Biotechnology Fundamentals (CRISPR, Genetics) courses to knowledge base.
        Includes 10 courses covering CRISPR gene editing, genetics, evolution, synthetic biology, and biotechnology.
        
        Returns:
            True if successful, False otherwise
        """
        biotechnology_courses = [
            {
                "title": "Industrial Biotechnology",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/industrial-biotech",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Industrial Biotechnology", "Biotechnology", "Biology", "Biotech"],
                "description": "Course covering industrial biotechnology including biotechnology"
            },
            {
                "title": "CRISPR: Gene Editing Applications",
                "provider": "edX",
                "url": "https://www.edx.org/course/crispr-gene-editing-applications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CRISPR", "Gene Editing", "Biotechnology", "Genetics"],
                "description": "Course covering CRISPR including gene editing applications"
            },
            {
                "title": "Introduction to Genetics and Evolution",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/genetics-evolution",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Genetics", "Evolution", "Biology", "Biotechnology"],
                "description": "Course covering introduction to genetics and evolution including biology"
            },
            {
                "title": "Gene Editing with CRISPR-Cas9",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/gene-editing-with-crispr-cas9-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["CRISPR-Cas9", "Gene Editing", "Biotechnology", "Genetics"],
                "description": "Course covering gene editing with CRISPR-Cas9 including biotechnology"
            },
            {
                "title": "Biotechnology: Antibodies & Protein Engineering",
                "provider": "edX",
                "url": "https://www.edx.org/course/biotechnology-antibodies-and-protein-engineering",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biotechnology", "Protein Engineering", "Antibodies", "Biotech"],
                "description": "Course covering biotechnology including antibodies and protein engineering"
            },
            {
                "title": "Stanford Introduction to Genetics and Evolution",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/genetics-evolution",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Genetics", "Evolution", "Biology", "Biotechnology"],
                "description": "Course covering Stanford introduction to genetics and evolution including biology"
            },
            {
                "title": "Principles of Synthetic Biology",
                "provider": "edX",
                "url": "https://www.edx.org/course/principles-of-synthetic-biology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Synthetic Biology", "Biotechnology", "Biology", "Biotech"],
                "description": "Course covering principles of synthetic biology including biotechnology"
            },
            {
                "title": "Modern Biology - Part 1 (Genetics)",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/modern-biology-1",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Genetics", "Modern Biology", "Biology", "Biotechnology"],
                "description": "Course covering modern biology part 1 including genetics"
            },
            {
                "title": "Understanding the Human Genome",
                "provider": "FutureLearn",
                "url": "https://www.futurelearn.com/courses/the-genomics-era",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Human Genome", "Genomics", "Genetics", "Biotechnology"],
                "description": "Course covering understanding the human genome including genomics"
            }
        ]
        
        log.info(f"📚 Adding {len(biotechnology_courses)} Biotechnology Fundamentals (CRISPR, Genetics) courses to knowledge base...")
        return self.batch_add_courses(biotechnology_courses)
    
    def add_pharmaceutical_management_courses(self) -> bool:
        """
        Add Pharmaceutical Management courses to knowledge base.
        Includes 10 courses covering drug discovery, development, commercialization, regulatory affairs, and pharmacovigilance.
        
        Returns:
            True if successful, False otherwise
        """
        pharmaceutical_courses = [
            {
                "title": "Drug Discovery, Development & Commercialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/drug-commercialization",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Drug Discovery", "Pharmaceutical", "Drug Development", "Pharma"],
                "description": "Course covering drug discovery, development and commercialization including pharmaceutical"
            },
            {
                "title": "Drug Development Product Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/drug-development-product-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Drug Development", "Product Management", "Pharmaceutical", "Pharma"],
                "description": "Specialization covering drug development product management including pharmaceutical"
            },
            {
                "title": "Medical Technology and Evaluation",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/medical-technology-evaluation",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Technology", "Pharmaceutical", "Healthcare Technology", "Medical Devices"],
                "description": "Course covering medical technology and evaluation including healthcare technology"
            },
            {
                "title": "The FDA and Prescription Drugs",
                "provider": "Coursera",
                "url": "https://www.edx.org/course/the-fda-and-prescription-drugs-current-controversies-in-context",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["FDA", "Pharmaceutical", "Regulatory Affairs", "Drug Regulation"],
                "description": "Course covering the FDA and prescription drugs including regulatory affairs"
            },
            {
                "title": "Good Clinical Practice (GCP)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/good-clinical-practice-ich-gcp/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["GCP", "Good Clinical Practice", "Clinical Trials", "Pharmaceutical"],
                "description": "Course covering good clinical practice including GCP and clinical trials"
            },
            {
                "title": "Pharmacovigilance & Drug Safety",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/pharmacovigilance-drug-safety-management/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pharmacovigilance", "Drug Safety", "Pharmaceutical", "Pharma"],
                "description": "Course covering pharmacovigilance and drug safety including pharmaceutical"
            },
            {
                "title": "Regulatory Affairs for Medical Devices & Pharmaceuticals",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/medical-device-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Regulatory Affairs", "Pharmaceutical", "Medical Devices", "FDA"],
                "description": "Specialization covering regulatory affairs for medical devices and pharmaceuticals"
            },
            {
                "title": "Clinical Data Management (CDM)",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/clinical-data-management-cdm-training-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Clinical Data Management", "CDM", "Clinical Trials", "Pharmaceutical"],
                "description": "Course covering clinical data management including CDM and clinical trials"
            },
            {
                "title": "Introduction to Pharmacy",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-to-pharmacy",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pharmacy", "Pharmaceutical", "Healthcare", "Medicine"],
                "description": "Course covering introduction to pharmacy including pharmaceutical"
            },
            {
                "title": "Pharma Mini-MBA: BioPharma Industry",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/biopharma-mini-mba/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pharma MBA", "BioPharma", "Pharmaceutical", "Business"],
                "description": "Course covering pharma mini-MBA including biopharma industry"
            }
        ]
        
        log.info(f"📚 Adding {len(pharmaceutical_courses)} Pharmaceutical Management courses to knowledge base...")
        return self.batch_add_courses(pharmaceutical_courses)
    
    def add_biomedical_engineering_courses(self) -> bool:
        """
        Add Biomedical Engineering courses to knowledge base.
        Includes 10 courses covering biomedical engineering, biomaterials, medical devices, bionics, and biomedical imaging.
        
        Returns:
            True if successful, False otherwise
        """
        biomedical_engineering_courses = [
            {
                "title": "Introduction to Biomedical Engineering",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/intro-biomedical-engineering",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biomedical Engineering", "Engineering", "Medical Engineering", "Healthcare Technology"],
                "description": "Course covering introduction to biomedical engineering including medical engineering"
            },
            {
                "title": "So You Want to Become a Biomedical Engineer",
                "provider": "edX",
                "url": "https://www.edx.org/course/so-you-want-to-become-a-biomedical-engineer",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biomedical Engineering", "Engineering", "Career", "Medical Engineering"],
                "description": "Course covering so you want to become a biomedical engineer including career"
            },
            {
                "title": "Biomaterials and Bioengineering",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/biomaterials",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biomaterials", "Bioengineering", "Biomedical Engineering", "Materials Science"],
                "description": "Course covering biomaterials and bioengineering including materials science"
            },
            {
                "title": "Introduction to Medical Software",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/introduction-medical-software",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Software", "Biomedical Engineering", "Healthcare Technology", "Software"],
                "description": "Course covering introduction to medical software including healthcare technology"
            },
            {
                "title": "Medical Device Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/medical-device-development",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Medical Device Development", "Biomedical Engineering", "Medical Devices", "Engineering"],
                "description": "Specialization covering medical device development including biomedical engineering"
            },
            {
                "title": "Bionics and Prosthetics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/bionics-prosthetics",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Bionics", "Prosthetics", "Biomedical Engineering", "Medical Devices"],
                "description": "Course covering bionics and prosthetics including biomedical engineering"
            },
            {
                "title": "Implantable Medical Devices",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/implantable-medical-devices",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Implantable Medical Devices", "Medical Devices", "Biomedical Engineering", "Engineering"],
                "description": "Course covering implantable medical devices including biomedical engineering"
            },
            {
                "title": "Biomedical Imaging",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/biomedical-imaging",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biomedical Imaging", "Medical Imaging", "Biomedical Engineering", "Healthcare Technology"],
                "description": "Course covering biomedical imaging including medical imaging"
            },
            {
                "title": "Nanotechnology and Nanosensors",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/nanotechnology",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Nanotechnology", "Nanosensors", "Biomedical Engineering", "Engineering"],
                "description": "Course covering nanotechnology and nanosensors including biomedical engineering"
            },
            {
                "title": "Fundamentals of Biomedical Engineering",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/fundamentals-of-biomedical-engineering/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Biomedical Engineering", "Fundamentals", "Engineering", "Medical Engineering"],
                "description": "Course covering fundamentals of biomedical engineering including medical engineering"
            }
        ]
        
        log.info(f"📚 Adding {len(biomedical_engineering_courses)} Biomedical Engineering courses to knowledge base...")
        return self.batch_add_courses(biomedical_engineering_courses)
    
    def add_pr_crisis_management_courses(self) -> bool:
        """
        Add Public Relations (PR) & Crisis Management courses to knowledge base.
        Includes 10 courses covering public relations, crisis management, reputation management, and media training.
        
        Returns:
            True if successful, False otherwise
        """
        pr_crisis_courses = [
            {
                "title": "Introduction to Public Relations and the Media",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/public-relations-media",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Relations", "PR", "Media Relations", "Communications"],
                "description": "Course covering introduction to public relations and the media including media relations"
            },
            {
                "title": "Public Relations for Digital Media",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/public-relations-digital-media",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Relations", "Digital Media", "PR", "Social Media"],
                "description": "Course covering public relations for digital media including social media"
            },
            {
                "title": "Crisis Management: Proseminar",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/crisis-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Crisis Management", "Public Relations", "Risk Management", "Communications"],
                "description": "Course covering crisis management proseminar including risk management"
            },
            {
                "title": "Reputation Management in a Digital World",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/reputation-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Reputation Management", "Public Relations", "Digital Marketing", "Brand Management"],
                "description": "Course covering reputation management in a digital world including brand management"
            },
            {
                "title": "Public Relations: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/public-relations-pr/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Relations", "PR", "Communications", "Media Relations"],
                "description": "Course covering public relations the complete guide including communications"
            },
            {
                "title": "Crisis Communications: Survival Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/crisis-communications-survival-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Crisis Communications", "Crisis Management", "Public Relations", "Communications"],
                "description": "Course covering crisis communications survival guide including crisis management"
            },
            {
                "title": "Integrated Marketing Communications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/integrated-marketing-communications",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Integrated Marketing Communications", "Marketing", "Public Relations", "Communications"],
                "description": "Course covering integrated marketing communications including marketing"
            },
            {
                "title": "Public Relations Foundations",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/public-relations-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Public Relations", "PR", "Communications", "Media Relations"],
                "description": "Course covering public relations foundations including communications"
            },
            {
                "title": "Media Training for Beginners",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/media-training-for-beginners-ace-your-media-interview/",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Media Training", "Public Relations", "Media Relations", "Communications"],
                "description": "Course covering media training for beginners including media relations"
            },
            {
                "title": "Working with the Media",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/working-with-media",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Media Relations", "Public Relations", "Communications", "PR"],
                "description": "Course covering working with the media including public relations"
            }
        ]
        
        log.info(f"📚 Adding {len(pr_crisis_courses)} Public Relations (PR) & Crisis Management courses to knowledge base...")
        return self.batch_add_courses(pr_crisis_courses)
    
    def add_journalism_digital_media_courses(self) -> bool:
        """
        Add Journalism & Digital Media courses to knowledge base.
        Includes 10 courses covering journalism, digital media, podcasting, mobile journalism, and storytelling.
        
        Returns:
            True if successful, False otherwise
        """
        journalism_courses = [
            {
                "title": "Become a Journalist: Report the News!",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/become-a-journalist",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Journalism", "News Reporting", "Writing", "Media"],
                "description": "Specialization covering become a journalist report the news including news reporting"
            },
            {
                "title": "English for Journalism",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/journalism",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Journalism", "English", "Writing", "News Reporting"],
                "description": "Course covering English for journalism including writing"
            },
            {
                "title": "The Complete Journalism Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/journalism-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Journalism", "News Reporting", "Writing", "Media"],
                "description": "Course covering the complete journalism course including news reporting"
            },
            {
                "title": "Podcast Masterclass: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/podcast-masterclass/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Podcasting", "Audio Production", "Journalism", "Digital Media"],
                "description": "Course covering podcast masterclass the complete guide including audio production"
            },
            {
                "title": "Journalism, the Future, and You",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/journalism-future",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Journalism", "Digital Media", "News Reporting", "Media"],
                "description": "Course covering journalism the future and you including digital media"
            },
            {
                "title": "Transmedia Storytelling: Narrative Worlds",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/transmedia-storytelling",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Transmedia Storytelling", "Storytelling", "Digital Media", "Content Creation"],
                "description": "Course covering transmedia storytelling narrative worlds including content creation"
            },
            {
                "title": "Digital Media and Truth",
                "provider": "edX",
                "url": "https://www.edx.org/course/digital-media-and-truth",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Digital Media", "Journalism", "Media Literacy", "News Reporting"],
                "description": "Course covering digital media and truth including media literacy"
            },
            {
                "title": "Good with Words: Writing and Editing",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/good-with-words",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Writing", "Editing", "Journalism", "Content Writing"],
                "description": "Specialization covering good with words writing and editing including content writing"
            },
            {
                "title": "Mobile Journalism: Smartphone Reporting",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mobile-journalism/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Mobile Journalism", "Journalism", "News Reporting", "Digital Media"],
                "description": "Course covering mobile journalism smartphone reporting including news reporting"
            },
            {
                "title": "Gathering and Developing the News",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/gathering-developing-news",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["News Reporting", "Journalism", "Investigative Journalism", "Media"],
                "description": "Course covering gathering and developing the news including investigative journalism"
            }
        ]
        
        log.info(f"📚 Adding {len(journalism_courses)} Journalism & Digital Media courses to knowledge base...")
        return self.batch_add_courses(journalism_courses)
    
    def add_event_planning_management_courses(self) -> bool:
        """
        Add Event Planning & Management courses to knowledge base.
        Includes 10 courses covering event planning, corporate events, wedding planning, sports events, and virtual events.
        
        Returns:
            True if successful, False otherwise
        """
        event_planning_courses = [
            {
                "title": "Successful Events: Event Planning and Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/successful-events-event-planning-management-marketing/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Event Planning", "Event Management", "Project Management", "Event Marketing"],
                "description": "Course covering successful events event planning and management including event marketing"
            },
            {
                "title": "Event Planning Foundations",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/event-planning-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Event Planning", "Event Management", "Project Management", "Event Coordination"],
                "description": "Course covering event planning foundations including event coordination"
            },
            {
                "title": "The Ultimate Guide to Professional Event Planning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/event-planning-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Event Planning", "Event Management", "Project Management", "Event Coordination"],
                "description": "Course covering the ultimate guide to professional event planning including event coordination"
            },
            {
                "title": "Wedding Planning: The Complete Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/wedding-planning-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Wedding Planning", "Event Planning", "Event Management", "Event Coordination"],
                "description": "Course covering wedding planning the complete guide including event coordination"
            },
            {
                "title": "International Entertainment and Sports Management",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/entertainment-sports-management",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sports Management", "Entertainment Management", "Event Management", "Event Planning"],
                "description": "Course covering international entertainment and sports management including event planning"
            },
            {
                "title": "Managing Major Sports Events",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/managing-major-sports-events",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sports Events", "Event Management", "Sports Management", "Event Planning"],
                "description": "Course covering managing major sports events including event planning"
            },
            {
                "title": "Corporate Event Planning",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/corporate-event-planning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Corporate Events", "Event Planning", "Event Management", "Business Events"],
                "description": "Course covering corporate event planning including business events"
            },
            {
                "title": "Virtual Event Planning & Management",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/virtual-event-planning/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Virtual Events", "Event Planning", "Event Management", "Digital Events"],
                "description": "Course covering virtual event planning and management including digital events"
            },
            {
                "title": "Project Management for Events",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/project-management-for-events/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Project Management", "Event Planning", "Event Management", "Event Coordination"],
                "description": "Course covering project management for events including event coordination"
            },
            {
                "title": "Introduction to Event Management",
                "provider": "Alison",
                "url": "https://alison.com/course/introduction-to-event-management",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Event Management", "Event Planning", "Event Coordination", "Project Management"],
                "description": "Course covering introduction to event management including event coordination"
            }
        ]
        
        log.info(f"📚 Adding {len(event_planning_courses)} Event Planning & Management courses to knowledge base...")
        return self.batch_add_courses(event_planning_courses)
    
    def add_audio_engineering_music_production_courses(self) -> bool:
        """
        Add Audio Engineering & Music Production courses to knowledge base.
        Includes 10 courses covering music production, audio engineering, sound design, film scoring, and music business.
        
        Returns:
            True if successful, False otherwise
        """
        audio_engineering_courses = [
            {
                "title": "Music Production Specialization",
                "provider": "Coursera",
                "url": "https://www.coursera.org/specializations/music-production",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Music Production", "Audio Engineering", "Sound Design", "Music"],
                "description": "Specialization covering music production including audio engineering and sound design"
            },
            {
                "title": "The Art of Music Production",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/producing-music",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Music Production", "Audio Engineering", "Sound Design", "Music"],
                "description": "Course covering the art of music production including audio engineering"
            },
            {
                "title": "Audio Signal Processing for Music Applications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/audio-signal-processing",
                "difficulty": "Advanced",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Audio Signal Processing", "Audio Engineering", "Music Production", "Sound Design"],
                "description": "Course covering audio signal processing for music applications including sound design"
            },
            {
                "title": "Music Production in Logic Pro X",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/music-production-in-logic-pro-x-course/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Logic Pro X", "Music Production", "Audio Engineering", "DAW"],
                "description": "Course covering music production in Logic Pro X including DAW"
            },
            {
                "title": "Hans Zimmer Teaches Film Scoring",
                "provider": "MasterClass",
                "url": "https://www.masterclass.com/classes/hans-zimmer-teaches-film-scoring",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Film Scoring", "Music Production", "Composition", "Audio Engineering"],
                "description": "Course covering Hans Zimmer teaches film scoring including composition"
            },
            {
                "title": "Audio Engineering: Mixing with Your Ears",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/mixing-with-your-ears/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Audio Engineering", "Mixing", "Music Production", "Sound Engineering"],
                "description": "Course covering audio engineering mixing with your ears including sound engineering"
            },
            {
                "title": "Pro Tools Basics",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/pro-tools-basics",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Pro Tools", "Audio Engineering", "DAW", "Music Production"],
                "description": "Course covering Pro Tools basics including DAW"
            },
            {
                "title": "Timbaland Teaches Producing and Beatmaking",
                "provider": "MasterClass",
                "url": "https://www.masterclass.com/classes/timbaland-teaches-producing-and-beatmaking",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Beatmaking", "Music Production", "Audio Engineering", "Hip Hop Production"],
                "description": "Course covering Timbaland teaches producing and beatmaking including hip hop production"
            },
            {
                "title": "Sound Design: The Ultimate Guide",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/sound-design-the-ultimate-guide/",
                "difficulty": "Intermediate",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Sound Design", "Audio Engineering", "Music Production", "Audio Design"],
                "description": "Course covering sound design the ultimate guide including audio design"
            },
            {
                "title": "Music Business Foundations",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/music-business-foundations",
                "difficulty": "Beginner",
                "duration": "Self-paced",
                "type": "course",
                "skills": ["Music Business", "Music Industry", "Music Production", "Entertainment Business"],
                "description": "Course covering music business foundations including entertainment business"
            }
        ]
        
        log.info(f"📚 Adding {len(audio_engineering_courses)} Audio Engineering & Music Production courses to knowledge base...")
        return self.batch_add_courses(audio_engineering_courses)



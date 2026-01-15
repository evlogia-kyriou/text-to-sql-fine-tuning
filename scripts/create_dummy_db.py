import sqlite3
import json
from pathlib import Path

def create_dummy_database():
    """Create a simple dummy database for testing"""
    
    # Create directory
    db_dir = Path('data/dummy')
    db_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = db_dir / 'company.sqlite'
    
    # Remove if exists
    if db_path.exists():
        db_path.unlink()
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        department_id INTEGER,
        salary REAL,
        hire_date TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT,
        budget REAL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department_id INTEGER,
        start_date TEXT,
        budget REAL,
        status TEXT
    )
    """)
    
    # Insert sample data
    cursor.executemany("""
    INSERT INTO departments (id, name, location, budget) 
    VALUES (?, ?, ?, ?)
    """, [
        (1, 'Engineering', 'San Francisco', 1000000),
        (2, 'Sales', 'New York', 500000),
        (3, 'Marketing', 'Los Angeles', 300000),
        (4, 'HR', 'Chicago', 200000)
    ])
    
    cursor.executemany("""
    INSERT INTO employees (id, name, age, department_id, salary, hire_date)
    VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (1, 'Alice Johnson', 32, 1, 120000, '2020-01-15'),
        (2, 'Bob Smith', 45, 1, 150000, '2018-03-20'),
        (3, 'Carol White', 28, 2, 80000, '2021-06-10'),
        (4, 'David Brown', 38, 2, 95000, '2019-09-05'),
        (5, 'Eve Davis', 29, 3, 70000, '2022-02-14'),
        (6, 'Frank Wilson', 41, 1, 130000, '2017-11-30'),
        (7, 'Grace Lee', 26, 3, 65000, '2023-01-20'),
        (8, 'Henry Martinez', 35, 4, 75000, '2020-08-25')
    ])
    
    cursor.executemany("""
    INSERT INTO projects (id, name, department_id, start_date, budget, status)
    VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (1, 'Website Redesign', 1, '2024-01-01', 50000, 'active'),
        (2, 'Mobile App', 1, '2024-02-15', 100000, 'active'),
        (3, 'Sales Campaign', 2, '2024-03-01', 30000, 'completed'),
        (4, 'Brand Refresh', 3, '2024-01-10', 40000, 'active'),
        (5, 'HR Portal', 4, '2023-12-01', 25000, 'completed')
    ])
    
    conn.commit()
    conn.close()
    
    print(f"✓ Dummy database created at: {db_path}")
    
    # Create schema info (BIRD format)
    schema = {
        "db_id": "company",
        "table_names": ["employees", "departments", "projects"],
        "column_names": [
            [-1, "*"],
            [0, "id"],
            [0, "name"],
            [0, "age"],
            [0, "department_id"],
            [0, "salary"],
            [0, "hire_date"],
            [1, "id"],
            [1, "name"],
            [1, "location"],
            [1, "budget"],
            [2, "id"],
            [2, "name"],
            [2, "department_id"],
            [2, "start_date"],
            [2, "budget"],
            [2, "status"]
        ],
        "column_types": [
            "text",
            "number", "text", "number", "number", "number", "text",
            "number", "text", "text", "number",
            "number", "text", "number", "text", "number", "text"
        ],
        "primary_keys": [1, 7, 11],
        "foreign_keys": [[4, 7], [13, 7]]
    }
    
    # Save schema
    with open(db_dir / 'schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"✓ Schema info saved to: {db_dir / 'schema.json'}")
    
    # Create test questions
    test_questions = [
        {
            "question": "What are all the employee names?",
            "evidence": "",
            "SQL": "SELECT name FROM employees",
            "difficulty": "easy"
        },
        {
            "question": "How many employees work in the Engineering department?",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Engineering'",
            "difficulty": "medium"
        },
        {
            "question": "What is the average salary of employees?",
            "evidence": "",
            "SQL": "SELECT AVG(salary) FROM employees",
            "difficulty": "easy"
        },
        {
            "question": "List all departments with their total employee count",
            "evidence": "",
            "SQL": "SELECT d.name, COUNT(e.id) FROM departments d LEFT JOIN employees e ON d.id = e.department_id GROUP BY d.name",
            "difficulty": "medium"
        },
        {
            "question": "Who is the highest paid employee?",
            "evidence": "",
            "SQL": "SELECT name FROM employees ORDER BY salary DESC LIMIT 1",
            "difficulty": "easy"
        },
        {
            "question": "What is the total budget of all active projects?",
            "evidence": "",
            "SQL": "SELECT SUM(budget) FROM projects WHERE status = 'active'",
            "difficulty": "easy"
        },
        {
            "question": "List all employees hired after 2020 with their department names",
            "evidence": "",
            "SQL": "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE e.hire_date > '2020-01-01'",
            "difficulty": "medium"
        },
        {
            "question": "Which department has the highest total salary cost?",
            "evidence": "",
            "SQL": "SELECT d.name FROM departments d JOIN employees e ON d.id = e.department_id GROUP BY d.name ORDER BY SUM(e.salary) DESC LIMIT 1",
            "difficulty": "hard"
        },
        {
            "question": "How many projects does each department have?",
            "evidence": "",
            "SQL": "SELECT d.name, COUNT(p.id) FROM departments d LEFT JOIN projects p ON d.id = p.department_id GROUP BY d.name",
            "difficulty": "medium"
        },
        {
            "question": "What is the name and salary of employees in San Francisco?",
            "evidence": "",
            "SQL": "SELECT e.name, e.salary FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.location = 'San Francisco'",
            "difficulty": "medium"
        }
    ]
    
    # Save test questions
    with open(db_dir / 'test_questions.json', 'w') as f:
        json.dump(test_questions, f, indent=2)
    
    print(f"✓ Test questions saved to: {db_dir / 'test_questions.json'}")
    print(f"\nCreated {len(test_questions)} test questions")
    
    return db_path, schema, test_questions


if __name__ == "__main__":
    print("="*80)
    print("CREATING DUMMY DATABASE")
    print("="*80)
    print()
    
    db_path, schema, questions = create_dummy_database()
    
    print("\n" + "="*80)
    print("DATABASE SUMMARY")
    print("="*80)
    print(f"\nTables: {', '.join(schema['table_names'])}")
    print(f"Total columns: {len(schema['column_names'])}")
    print(f"Test questions: {len(questions)}")
    
    # Show sample data
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\nSample data from employees:")
    cursor.execute("SELECT name, age, salary FROM employees LIMIT 3")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()
    
    print("\n✅ Ready for testing!")
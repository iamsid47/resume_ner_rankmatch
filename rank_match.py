from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy

def rank_cvs_with_transformers(job_description, cv_ner_outputs_list):
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    job_description_embedding = model.encode(job_description)

    ranked_cvs = []
    for idx, cv in enumerate(cv_ner_outputs_list):
        cv_text = ' '.join([token for token, _ in cv])
        cv_embedding = model.encode(cv_text)

        cosine_similarity = util.pytorch_cos_sim(job_description_embedding, cv_embedding).numpy()[0][0]
        ranked_cvs.append((idx, cosine_similarity))

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)
    return ranked_cvs

def rank_cvs_with_spacy_similarity(job_description, cv_text_list):
    nlp = spacy.load('en_core_web_lg')

    jd_doc = nlp(job_description)

    ranked_cvs = []
    for idx, cv_text in enumerate(cv_text_list):
        cv_doc = nlp(cv_text)

        similarity_score = jd_doc.similarity(cv_doc)
        ranked_cvs.append((idx, similarity_score))

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)
    return ranked_cvs




# job_description = '''
# **Job Title:** Senior Software Engineer

# **Company:** InnovateTech Solutions

# **Location:** San Francisco, CA

# **Job Type:** Full-Time

# **Industry:** Technology

# **Job ID:** IT12345

# **Application Deadline:** September 15, 20XX

# **Company Overview:**
# At InnovateTech Solutions, we're at the forefront of cutting-edge technology, driving innovation and pushing boundaries. Our mission is to create transformative software solutions that shape industries and improve lives. Join our dynamic team and be part of the future.

# **Position Overview:**
# As a Senior Software Engineer, you will play a pivotal role in designing, developing, and delivering software solutions that solve complex challenges. You will collaborate with cross-functional teams to create scalable and robust applications that drive our business forward.

# **Responsibilities:**
# - Lead the design and implementation of software applications using a variety of programming languages and technologies.
# - Collaborate with product managers, designers, and fellow engineers to define project requirements and specifications.
# - Mentor and guide junior engineers, fostering a culture of continuous learning and professional growth.
# - Perform code reviews and ensure code quality, performance, and security standards are met.
# - Contribute to architectural decisions and technology selection to ensure scalability and maintainability.
# - Troubleshoot and debug complex technical issues, ensuring timely resolution and minimal impact on users.
# - Stay up-to-date with industry trends and emerging technologies to drive innovation.

# **Qualifications:**
# - Bachelor's or Master's degree in Computer Science or related field.
# - 5+ years of professional software development experience, with a focus on full-stack development.
# - Proficiency in languages such as Python, Java, or C# and modern front-end frameworks (React, Angular, or Vue.js).
# - Strong experience with database design, SQL, and NoSQL databases.
# - Solid understanding of software architecture and design patterns.
# - Excellent problem-solving skills and a passion for delivering high-quality solutions.
# - Effective communication and collaboration skills to work within cross-functional teams.
# - Experience with cloud technologies (AWS, Azure, or GCP) is a plus.

# **Salary and Benefits:**
# - Competitive salary and performance-based bonuses.
# - Comprehensive benefits package including health, dental, and vision insurance.
# - 401(k) retirement plan with company match.
# - Flexible work arrangements, including remote options.
# - Professional development and training opportunities.
# - Vibrant and inclusive work culture.

# **Application Instructions:**
# To apply, please submit your resume, cover letter, and a link to your portfolio showcasing relevant projects. Click the link below to apply directly through our online portal. For any inquiries, contact our HR department at careers@innovatetech.com.

# **Equal Opportunity Employer:**
# InnovateTech Solutions is an equal opportunity employer. We celebrate diversity and are committed to creating an inclusive environment for all employees.

# Learn more about us at www.innovatetech.com.
# '''

# cv_ner_outputs_list = [
#    [ ['www.linkedin.com/in/usman-alib5173a235', 'LINKEDIN '],['Negotiation\nElectric Motors\n\n', 'EXPERIENCE'],['USMAN Ali\n\n', 'ORGANIZATION'],['Electrical Maintinace Technician', 'TITLE-JOB'],['Al Khobar', 'SKILL'],['Eastern', 'LOCATION'],['Saudi Arabia\n\n', 'LOCATION'],['advice', 'SKILL'],['SRACO Human Resources Company', 'EXPERIENCE'],['Electrical Maintanice Technician .', 'EXPERIENCE'],['Electrical Technician', 'EDUCATION'],['September 2019 - Present', 'DATE_RANGE '],['4 years', 'DATE'],['Al Jubayl', 'EDUCATION'],['Saudi Arabia\n', 'LOCATION'],['Electrical Technician', 'TITLE-JOB'],['June 2023', 'DATE'] ],
#    [['www.linkedin.com/in/johndoe', 'LINKEDIN'], ['Software Engineering\nPython\nJava\nJavaScript\n', 'EXPERIENCE'], ['John Doe', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['June 2018 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/janedoe', 'LINKEDIN'], ['Data Analysis\nMachine Learning\nSQL\n', 'EXPERIENCE'], ['Jane Doe', 'NAME'], ['Data Scientist', 'TITLE-JOB'], ['Statistics', 'EDUCATION'], ['August 2017 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/johncarter', 'LINKEDIN'], ['Project Management\nAgile\nScrum\n', 'EXPERIENCE'], ['John Carter', 'NAME'], ['Project Manager', 'TITLE-JOB'], ['Business Administration', 'EDUCATION'], ['May 2015 - Present', 'DATE_RANGE']],
#       [['www.linkedin.com/in/akashgupta', 'LINKEDIN'], ['Software Development\nJava\nSpring Framework\n', 'EXPERIENCE'], ['Akash Gupta', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['January 2017 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/priyasharma', 'LINKEDIN'], ['Marketing Strategies\nDigital Marketing\nSEO\n', 'EXPERIENCE'], ['Priya Sharma', 'NAME'], ['Digital Marketing Manager', 'TITLE-JOB'], ['Marketing Management', 'EDUCATION'], ['May 2016 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/rahuljain', 'LINKEDIN'], ['Project Management\nAgile\nScrum\n', 'EXPERIENCE'], ['Rahul Jain', 'NAME'], ['Project Manager', 'TITLE-JOB'], ['Business Administration', 'EDUCATION'], ['October 2014 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/nehasingh', 'LINKEDIN'], ['Data Analysis\nMachine Learning\nPython\n', 'EXPERIENCE'], ['Neha Singh', 'NAME'], ['Data Scientist', 'TITLE-JOB'], ['Statistics', 'EDUCATION'], ['July 2015 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/amitpatel', 'LINKEDIN'], ['Finance Management\nBudgeting\nFinancial Planning\n', 'EXPERIENCE'], ['Amit Patel', 'NAME'], ['Financial Analyst', 'TITLE-JOB'], ['Finance', 'EDUCATION'], ['September 2018 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/anitakumar', 'LINKEDIN'], ['Human Resources\nEmployee Relations\nRecruitment\n', 'EXPERIENCE'], ['Anita Kumar', 'NAME'], ['HR Manager', 'TITLE-JOB'], ['Human Resource Management', 'EDUCATION'], ['June 2013 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/manishverma', 'LINKEDIN'], ['Software Development\nPython\nDjango\nFlask\n', 'EXPERIENCE'], ['Manish Verma', 'NAME'], ['Software Developer', 'TITLE-JOB'], ['Computer Engineering', 'EDUCATION'], ['March 2016 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/marcusmiller', 'LINKEDIN'], ['Software Engineering\nJava\nSpring Framework\n', 'EXPERIENCE'], ['Marcus Miller', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['February 2016 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/lauraschmidt', 'LINKEDIN'], ['Software Development\nJava\nSpring Boot\n', 'EXPERIENCE'], ['Laura Schmidt', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Engineering', 'EDUCATION'], ['June 2017 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/thomasandersson', 'LINKEDIN'], ['Software Engineering\nPython\nDjango\n', 'EXPERIENCE'], ['Thomas Andersson', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['May 2015 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/sophiewagner', 'LINKEDIN'], ['Software Development\nJava\nSpring Framework\n', 'EXPERIENCE'], ['Sophie Wagner', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['April 2016 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/robertomartinez', 'LINKEDIN'], ['Software Engineering\nPython\nDjango\n', 'EXPERIENCE'], ['Roberto Martinez', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['July 2014 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/emilielarsson', 'LINKEDIN'], ['Software Development\nPython\nFlask\n', 'EXPERIENCE'], ['Emilie Larsson', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Engineering', 'EDUCATION'], ['August 2016 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/oliverjackson', 'LINKEDIN'], ['Software Engineering\nJava\nSpring Boot\n', 'EXPERIENCE'], ['Oliver Jackson', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['March 2017 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/claudiaklein', 'LINKEDIN'], ['Software Development\nPython\nDjango\n', 'EXPERIENCE'], ['Claudia Klein', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['October 2015 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/andreaschneider', 'LINKEDIN'], ['Software Development\nJava\nSpring Framework\n', 'EXPERIENCE'], ['Andrea Schneider', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Science', 'EDUCATION'], ['January 2018 - Present', 'DATE_RANGE']],
#     [['www.linkedin.com/in/louisemartin', 'LINKEDIN'], ['Software Development\nPython\nFlask\n', 'EXPERIENCE'], ['Louise Martin', 'NAME'], ['Software Engineer', 'TITLE-JOB'], ['Computer Engineering', 'EDUCATION'], ['September 2016 - Present', 'DATE_RANGE']]
   
# ]

# ranked_cvs = rank_cvs_with_transformers(job_description, cv_ner_outputs_list)

# for idx, (cv_idx, score) in enumerate(ranked_cvs, start=1):
#     print(f"Rank {idx}: CV {cv_idx} - Cosine Similarity Score: {score:.4f}")

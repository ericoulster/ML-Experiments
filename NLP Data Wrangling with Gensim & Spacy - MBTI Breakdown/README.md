Data for this project came from https://www.kaggle.com/datasnaek/mbti-type - it was too large to include in this repo.

All processing which took place on the data for analysis is contained within the doc.

While most of my projects start with a specific problem, in this case I wanted to explore data about the Myers-Briggs personality inventory

The source data is from a forum of personality psychology enthusiasts called 'Personality Cafe'. It is forum posts from specific people, along with their Myers-Briggs category.

Data was partitioned into it's 4 set of binaries (introvert-extrovert, intuitive-sensing, feeling-thinking, judging-perceiving), then descriptive statistics were provided using tf-idf vectors and descriptive statistics.

This is technically more of a 'descriptive statistical' project than a machine learning model, largely because the data wasn't well-suited to analysis.

Descriptive statistical methods along with attempts at modelling found the data was too similar between personality types to warrant significant results.

Nevertheless, the minute differences between these 4 sets of personality binaries have been catalogued, along with associated wordclouds for each binary.

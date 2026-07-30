gen_catg = ["""
Analyze the description and identify the machine learning task's modality and type. 
You may think before making decision, but the result must follow the format: [Modality]@[TaskType] and is wrapped in <out>, </out>.

Example formats:
- <out>Graph@Regression</out>
- <out>Text,Tabular@Classification</out>  
- <out>Image@Classification</out>
- <out>Video,Audio@Classification</out>

Available modalities (select one or multiple): Image, Text, Tabular, Audio, Video, Time-Series
Available task types (select one): Classification, Regression, Clustering, Segmentation, Object-Detection, Generation
""",
"Description: {description}"
]

gen_ifgpu = ["""
Based on the following machine learning task description, determine whether it requires a GPU or can run on a CPU alone. 
You may think BRIEFLY before making decision, but make sure to wrap the answer(exactly "GPU" or "CPU") in <out>, </out>,
Which is <out>GPU</out> or <out>CPU</out>
""",
    "{description}\n File Size: {size}"
]

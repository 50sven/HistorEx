# "Towards Natural Lanuage Understanding"

![alt text](https://github.com/50sven/ISE/blob/master/Repository_image.png)

This repository contains a pipeline for training document/word vectors, extracting/linking named entities and an interactive visualization tool to explore the findings. 

	@article{tnlu, 
		title={Towards Natural Lanuage Understanding}, 
		author={Brunzel, Michael and Mueller, Sven and Kaun, Daniela}, 
		year={2019}, 
	}


## Getting started
All code was written in Python Version 3.6. Requirements are listed in requirements.txt. To get started use:

	git clone https://github.com/50sven/ISE
    cd ISE
    pip install -r requirements.txt
    
Doc2Vec:
	* produce raw text data (`Doc2Vec_Preprocessing.py`)
	* train document and word embeddings (`Doc2Vec_Training.py`)
	* Evaluate embeddings (`Doc2Vec_Evaluation.py`, `WordVector_Evaluation.py`)
	
Entities:
	* Parse html/xml files (`parsing.py`)
	* Extract entities by tag or via spacy (`extract_information.py`)
	* Get data from wikidata (`Link_wikidata.py`)
	
Dashboard:
	* Screencast -> Screencast_Dashboard.mp4
	* use given data or produce own data (`assets`)
	* run app on localhost (`run_app.py`)


# Contributors

[Daniela Kaun](https://github.com/dakaun), [Michael Brunzel](https://github.com/michael-brunzel), [Sven Mueller](https://github.com/50sven)


# License

The MIT License ([MIT](http://opensource.org/licenses/mit-license.php))

Copyright (c) 2018 "Names of contributors"

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

echo "install python dependencies"
easy_install --upgrade six
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
pip install -r requirements.txt

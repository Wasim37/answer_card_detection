'''
Author: wangxin
Date: 2021-05-25 12:12:52
LastEditTime: 2021-05-28 17:54:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''

from flask import Flask

app = Flask(__name__)
@app.route('/detection', methods=["POST"])
def answer_card_detection():
    return "get_answer_from_sheet"


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890, debug=True)
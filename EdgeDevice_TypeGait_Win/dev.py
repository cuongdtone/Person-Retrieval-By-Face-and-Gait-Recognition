# # -*- coding: utf-8 -*-
# # @Organization  : TMT
# # @Author        : Cuong Tran
# # @Time          : 11/25/2022
#
# import datetime
# from flask import Flask, render_template_string
#
# app = Flask(__name__)
#
# @app.route("/")
# def index():
#     return render_template_string("""<!DOCTYPE html>
# <html>
#
# <head>
# <meta charset="utf-8" />
# <title>Test</title>
#
# <script type="text/javascript" src="http://code.jquery.com/jquery-1.8.0.min.js"></script>
#
# <script type="text/javascript">
# function updater() {
#   $.get('/data', function(data) {
#     $('#time').html(data);  // update page with new data
#   });
# };
#
# setInterval(updater, 1000);  // run `updater()` every 1000ms (1s)
# </script>
#
# </head>
#
# <body>
# Date & Time: <span id="time"><span>
# </body>
#
# </html>""")
#
#
# @app.route('/data')
# def data():
#     """send current content"""
#     return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
# if __name__ == "__main__":
#     app.run(debug=False)
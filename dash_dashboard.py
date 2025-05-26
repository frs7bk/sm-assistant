
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# تطبيق Dash
app = dash.Dash(__name__)
server = app.server

# بيانات وهمية مؤقتة
moods = ["سعيد", "محايد", "حزين", "غاضب"]
values = [15, 30, 10, 5]

df = pd.DataFrame({"المزاج": moods, "عدد المرات": values})

fig = px.bar(df, x="المزاج", y="عدد المرات", title="تحليل المشاعر الصوتية")

# تصميم الصفحة
app.layout = html.Div([
    html.H1("لوحة تحكم المساعد الذكي", style={"textAlign": "center", "color": "#2c3e50"}),
    dcc.Graph(figure=fig),
    html.Div("مرحبًا بك في نظام Fvisuals Assistant الذكي", style={"marginTop": 30, "fontSize": 18}),
])

if __name__ == "__main__":
    app.run_server(debug=True)

from plotly import graph_objects as go


def update_layout(fig: go.Figure, title: str, x_title: str = 'X', y_title: str = 'Y'):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        font=dict(
            family="Courier New, monospace",
            size=11,
            color="RebeccaPurple"
        ),
        height=700,
    )

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv(r"C:\Users\narni\Downloads\tumor_growth_data.csv")
df = df[["Tumor Type", "k_p (mmHg/day)"]].dropna()
tumor_types = df["Tumor Type"].unique()
growth_rates = {row["Tumor Type"]: abs(row["k_p (mmHg/day)"]) * 100 for _, row in df.iterrows()}

# Constants
grid_size = 50
hypoxia_threshold = 0.5
necrosis_threshold = 0.2

def simulate_tumor_growth(rate, num_steps):
    grid = np.zeros((grid_size, grid_size))
    grid[grid_size//2, grid_size//2] = 1
    oxygen = np.ones((grid_size, grid_size))
    
    for _ in range(num_steps):
        new_grid = grid.copy()
        oxygen[:, 0] = oxygen[:, -1] = oxygen[0, :] = oxygen[-1, :] = 1
        oxygen = (oxygen +
                  np.roll(oxygen, 1, 0) +
                  np.roll(oxygen, -1, 0) +
                  np.roll(oxygen, 1, 1) +
                  np.roll(oxygen, -1, 1)) / 5

        for x in range(1, grid_size-1):
            for y in range(1, grid_size-1):
                if grid[x, y] > 0:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if grid[x+dx, y+dy] == 0 and np.random.rand() < rate:
                            new_grid[x+dx, y+dy] = 1
                    oxygen[x, y] -= 0.005
                    oxygen[x, y] = max(0, oxygen[x, y])
                    if oxygen[x, y] < necrosis_threshold:
                        new_grid[x, y] = 0
                    elif oxygen[x, y] < hypoxia_threshold:
                        new_grid[x, y] = 0.5
        grid = new_grid
    return grid

def generate_2d_heatmap(grid):
    return go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=[
            (0.0, "purple"),   # Necrotic (0)
            (0.5, "green"),    # Hypoxic (0.5)
            (1.0, "yellow")    # Viable (1.0)
        ],
        zmin=0, zmax=1,
        showscale=False
    )).update_layout(
        title="2D Tumor Growth Simulation",
        margin=dict(t=30, l=0, r=0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

def generate_3d_surface(grid):
    return go.Figure(data=[go.Surface(
        z=grid,
        colorscale=[
            (0.0, "purple"),   # Dead
            (0.5, "green"),    # Hypoxic
            (1.0, "yellow")    # Viable
        ],
        showscale=False
    )]).update_layout(
        title="3D Tumor Growth Visualization",
        margin=dict(t=30, l=0, r=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

# Dash App
app = dash.Dash(__name__)
app.title = "Tumor Growth Simulation"

app.layout = html.Div([
    html.H1("Interactive Tumor Growth Dashboard"),
    
    html.Label("Select Tumor Type:"),
    dcc.Dropdown(id="tumor-selector", options=[{"label": t, "value": t} for t in tumor_types], value=tumor_types[0]),
    
    html.Label("Simulation Timesteps:"),
    dcc.Slider(
        id="step-slider",
        min=10,
        max=200,
        step=10,
        value=60,
        marks={i: f"{i}" for i in range(10, 201, 20)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Div([
        dcc.Graph(id="tumor-2d", style={"display": "inline-block", "width": "49%"}),
        dcc.Graph(id="tumor-3d", style={"display": "inline-block", "width": "49%"})
    ]),
    
    html.Div([
        html.H4("Color Legend:"),
        html.Ul([
            html.Li("🟣 Purple = Necrotic (Dead Cells)"),
            html.Li("🟢 Green = Hypoxic (Low Oxygen)"),
            html.Li("🟡 Yellow = Viable Tumor Cells"),
        ])
    ], style={"marginTop": "20px"})
])

@app.callback(
    [Output("tumor-2d", "figure"), Output("tumor-3d", "figure")],
    [Input("tumor-selector", "value"), Input("step-slider", "value")]
)
def update_plots(tumor_type, num_steps):
    rate = growth_rates.get(tumor_type, 0.001)
    grid = simulate_tumor_growth(rate, num_steps)
    return generate_2d_heatmap(grid), generate_3d_surface(grid)

if __name__ == '__main__':
    app.run(debug=True)
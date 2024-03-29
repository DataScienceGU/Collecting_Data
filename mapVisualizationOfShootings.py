import plotly.graph_objects as go
import chart_studio.plotly as py
import pandas as pd


def createFigure(myDataFrame):
    myDataFrame['text'] = myDataFrame['case'] + \
        '<br>Total Victims: ' + (myDataFrame['total_victims']).astype(str)
    limits = [(0, 6), (7, 15), (16, 605)]
    colors = ["royalblue", "crimson", "lightseagreen"]

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=myDataFrame['longitude'],
        lat=myDataFrame['latitude'],
        text=myDataFrame['text'],
        marker=dict(
            size=myDataFrame['total_victims'],
            color=colors[0],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode='area'),
        name="mass shootings by location and number of victims"
    )
    )

    fig.update_layout(
        title_text='Mass Shootings Corresponding to Total Victims Across the United States',
        showlegend=True,
        geo=dict(
            scope='usa',
            landcolor='rgb(217, 217, 217)',
        ),
    )

    py.plot(fig, filename="map_of_mass_shootings.html", auto_open=True)

    fig.show()


def readData(filename):
    # read the csv data and return a pandas dataframe
    myDataFrame = pd.read_csv(
        filename, sep=",", encoding='latin1')
    return myDataFrame


def main():
    filename = "mass_shootings.csv"
    myDataFrame = readData(filename)
    createFigure(myDataFrame)


if __name__ == "__main__":
    main()

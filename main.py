import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

    def clean_data(self, df):
        cols_to_drop = ['Title', 'Review Text', 'Clothing ID']
        return df.dropna().drop(cols_to_drop, axis=1)


class DataAnalyzer:

    def get_average_of_column(self, df):
        return df.groupby('Rating')['Positive Feedback Count'].mean()

    def get_distribution_of_column(self, df):
        return df['Rating'].value_counts(normalize=True)

    def get_median_of_column(self, df):
        return df.groupby('Recommended IND')['Age'].median()

    def get_mode_of_column(self, df):
        return df.groupby('Department Name')['Positive Feedback Count'].apply(lambda x: x.mode())


class DataVisualizer:

    def plot_line_chart(self, df, col_name):
        plt.plot(df[col_name])
        plt.xlabel('Index')
        plt.ylabel(col_name)
        plt.title('Line Chart of {}'.format(col_name))
        plt.show()

    def plot_distribution(self, df, col_name):
        plt.hist(df[col_name], bins=10, color='green')
        plt.xlabel(col_name)
        plt.ylabel('Frequency')
        plt.title('Histogram of {}'.format(col_name))
        plt.show()

    def plot_pie(self, df, col_name):
        counts = df[col_name].value_counts()
        plt.pie(counts.values, labels=counts.index.tolist())
        plt.title('Pie Chart of {}'.format(col_name))
        plt.show()

    def plot_scatter(self, df, col_name):
        plt.scatter(np.arange(len(df)), df[col_name])
        plt.xlabel('Index')
        plt.ylabel(col_name)
        plt.title('Scatter Plot of {}'.format(col_name))
        plt.show()


# Load and clean the data
processor = DataProcessor('D:\Downloads2\data\Python-Project\Womens Clothing E-Commerce Reviews.csv')
data = processor.load_data()
clean_data = processor.clean_data(data)

# Analyze the data
analyzer = DataAnalyzer()
average_feedback = analyzer.get_average_of_column(clean_data)
rating_distribution = analyzer.get_distribution_of_column(clean_data)
median_age = analyzer.get_median_of_column(clean_data)
positive_feedback_mode = analyzer.get_mode_of_column(clean_data)

# Visualize the data
visualizer = DataVisualizer()
visualizer.plot_line_chart(clean_data, 'Rating')
visualizer.plot_distribution(clean_data, 'Age')
visualizer.plot_pie(clean_data, 'Recommended IND')
visualizer.plot_scatter(clean_data, 'Positive Feedback Count')

# Save the visualizations to the output folder
plt.savefig('output/line_chart.png')
plt.savefig('output/histogram.png')
plt.savefig('output/pie_chart.png')
plt.savefig('output/scatter_plot.png')

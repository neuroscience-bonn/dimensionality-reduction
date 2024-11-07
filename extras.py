def bar_graph(values):
    """
    Generate a simple bar graph from a list of values. The graph is a string
    consisting of block characters, where the height of each block is proportional
    to the corresponding value.

    Parameters
    ----------
    values : list of numbers
        The values to be graphed.

    Returns
    -------
    graph : str
        The bar graph as a string.
    """
    chars = [' ' ,'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']  # Define the block characters
    graph = ''  # Initialize the graph string
    max_height = max(values)  # Find the maximum value to scale the graph
    for value in values:
        height = int(8 * value / max_height)  # Convert the value to a height
        bar = chars[height]  # Choose the right bar character
        graph += bar + ' '  # Add the bar to the graph string
    return graph
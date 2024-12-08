import streamlit as st
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import os
import json
from openai import OpenAI
import platform
import psutil
import requests
from collections import defaultdict

conn = sqlite3.connect('anomalies.db')
cursor = conn.cursor()

# Point to the local server
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="lm-studio")
model = "lmstudio-community-beta/Beta/Llama-3.2-3B-QNN"

GOOGLE_SEARCH_API_KEY = "AIzaSyBo9lCM5SC8gPJulp6y_rYMoV0loprW_jM"
SEARCH_ENGINE_ID = "b08c69d5b45c8421a"

system_info = {
    "CPU": platform.processor(),
    "Cores": psutil.cpu_count(logical=False),
    "Logical Cores": psutil.cpu_count(logical=True),
    "Memory": f"{round(psutil.virtual_memory().total / 1e9, 2)} GB",
    "Disk": f"{round(psutil.disk_usage('/').total / 1e9, 2)} GB",
    "OS": platform.system(),
    "OS Version": platform.version(),
    "Architecture": platform.architecture()[0],
}

def fetch_logs(incident_time):
    """
    Fetch log lines within a 2-hour range around the given incident time.

    :param incident_time: The incident time as a datetime object.
    :return: A list of log lines within the time range.
    """

    # Calculate the time range
    start_time = incident_time - timedelta(hours=2)
    start_time_num = int(start_time.timestamp())
    end_time = incident_time + timedelta(hours=2)

    # Query to fetch the log lines within the time range
    query = '''
        SELECT timestamp, log_level, component, message, template, log_line, cpu_usage, memory_usage
        FROM anomalies
        WHERE incident_time BETWEEN ? AND ?
        ORDER BY incident_time
    '''

    # Execute the query
    cursor.execute(query, (start_time, end_time))
    rows = cursor.fetchall()

    # Convert the data to a list of dictionaries
    log_lines = []
    for row in rows:
        log_lines.append({
            "timestamp": row[0],
            "log_level": row[1],
            "component": row[2],
            "message": row[3],
            "template": row[4],
            "log_line": row[5],
            "cpu_usage": row[6],
            "memory_usage": row[7]
        })

    # Convert timestamps and group by minute
    cpu_usage_by_minute = defaultdict(list)
    memory_usage_by_minute = defaultdict(list)

    for row in rows:
        timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        minute_key = timestamp.replace(second=0)  # Strip seconds for grouping
        cpu_usage_by_minute[minute_key].append(row[6])  # CPU Usage
        memory_usage_by_minute[minute_key].append(row[7])  # Memory Usage

    # Calculate averages
    cpu_usage_averages = {
        key.strftime("%Y-%m-%d %H:%M:%S"): sum(values) / len(values)
        for key, values in cpu_usage_by_minute.items()
    }

    memory_usage_averages = {
        key.strftime("%Y-%m-%d %H:%M:%S"): sum(values) / len(values)
        for key, values in memory_usage_by_minute.items()
    }

    logs = [log["log_line"] for log in log_lines]

    return logs, cpu_usage_averages, memory_usage_averages

# Function to generate data for synthetic usage plots
def generate_usage_plot(cpu_points, memory_points, combined=False):
    fig = go.Figure()

    if cpu_points:
        # Extract times and values for CPU
        cpu_times = list(cpu_points.keys())
        cpu_values = list(cpu_points.values())

        # Create an array of all minutes (0 to 120)
        full_times = np.arange(0, 121, 1)

        # Perform cubic interpolation for CPU
        cpu_cubic_interp = interp1d(cpu_times, cpu_values, kind='cubic')
        cpu_full_values_cubic = cpu_cubic_interp(full_times)

        # Add CPU usage cubic interpolation line
        fig.add_trace(go.Scatter(
            x=full_times, 
            y=cpu_full_values_cubic, 
            mode='lines', 
            name='CPU Usage',
            line=dict(color='blue')
        ))

        # Add intermediate points for CPU
        fig.add_trace(go.Scatter(
            x=cpu_times, 
            y=cpu_values, 
            mode='markers', 
            name='Critical Events',
            marker=dict(color='red', size=8)
        ))

    if memory_points:
        # Extract times and values for Memory
        memory_times = list(memory_points.keys())
        memory_values = list(memory_points.values())

        # Create an array of all minutes (0 to 120)
        full_times = np.arange(0, 121, 1)

        # Perform cubic interpolation for Memory
        memory_cubic_interp = interp1d(memory_times, memory_values, kind='cubic')
        memory_full_values_cubic = memory_cubic_interp(full_times)

        # Add Memory usage cubic interpolation line
        fig.add_trace(go.Scatter(
            x=full_times, 
            y=memory_full_values_cubic, 
            mode='lines', 
            name='Memory Usage',
            line=dict(color='green')
        ))

        # Add intermediate points for Memory
        fig.add_trace(go.Scatter(
            x=memory_times, 
            y=memory_values, 
            mode='markers', 
            name='Critical Events',
            marker=dict(color='purple', size=8)
        ))

    # Update layout
    title = "CPU and Memory Usage Data" if combined else "Usage Data"
    fig.update_layout(
        title=title,
        xaxis_title="Time (minutes)",
        yaxis_title="Usage (%)",
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

# Streamlit Chatbot
st.title("cOS - Pilot")

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": " Hello! I'm your OS Co-Pilot, here to make your digital life smoother and more productive. Whether you need insights into your system's performance, help debugging issues, or assistance with everyday tasks, I've got you covered. Just let me know how I can assist you today!"}]

# Initialize session state for date and time if not set
if "selected_date" not in st.session_state:
    st.session_state.selected_date = datetime.now().date()

if "selected_time" not in st.session_state:
    st.session_state.selected_time = datetime.now().time()

# Date picker
st.session_state.selected_date = st.date_input(
    "Select a date:",
    value=st.session_state.selected_date
)

# Time picker
st.session_state.selected_time = st.time_input(
    "Select a time:",
    value=st.session_state.selected_time
)

# Combine date and time into a single datetime object
selected_datetime = datetime.combine(
    st.session_state.selected_date, st.session_state.selected_time
)

# Format the datetime in the desired format: YYYY-MM-DD HH:MM:SS
formatted_datetime = selected_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Display the formatted date and time
st.write("Analysing your system at:", formatted_datetime)

# Example usage
incident_time = datetime.strptime(formatted_datetime, '%Y-%m-%d %H:%M:%S')
logs, cpu_data, memory_data = fetch_logs(incident_time)

# Display conversation
# for msg in st.session_state.messages:
#     if msg["role"] == "assistant":
#         st.markdown(f"**Assistant:** {msg['content']}")
#     else:
#         st.markdown(f"**You:** {msg['content']}")

# User input
user_input = st.text_input("You:", key="input")

messages = [
        {
            "role": "system",
            "content": "You are the OS Co-Pilot, a versatile and intelligent assistant designed to enhance user productivity and system efficiency. You provide detailed insights into system hardware, correlate performance metrics like CPU and memory usage with patterns in logs to identify anomalies, and proactively assist with system maintenance and debugging. From automating routine tasks and troubleshooting issues to performing web searches for unknown queries, managing files, optimizing workflows, and providing contextual recommendations, you are the ultimate partner in helping users navigate and master their digital environment.",
        }]

def format_input_query(logs, cpu_data, memory_data, user_input):
    log_data = "\n".join(logs)
    cpu_data_str = '\n'.join([f"{key}: {value}%" for key, value in cpu_data.items()])
    memory_data_str = '\n'.join([f"{key}: {value}%" for key, value in memory_data.items()])
    prompt = f"""
Here are the error logs and relavent data:
{log_data}

Here is the CPU usage data:
{cpu_data_str}

Here is the memory usage data:
{memory_data_str}

Here is the user query:
{user_input}

Based the log_data provided, analyse the user query and provide appropriate response.
"""
    

    return prompt

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Define intermediate points for CPU and Memory usage
    cpu_points = {0: 10, 30: 50, 60: 30, 90: 70, 120: 20}
    memory_points = {0: 30, 30: 70, 60: 50, 90: 80, 120: 40}

    # Check user input for plots
    if "cpu" in user_input.lower() and "memory" in user_input.lower():
        st.session_state.messages.append({"role": "assistant", "content": "Here's the combined CPU and Memory usage plot!"})
        # Display combined CPU and Memory usage plot
        st.plotly_chart(generate_usage_plot(cpu_points, memory_points, combined=True))
    elif "cpu" in user_input.lower():
        st.session_state.messages.append({"role": "assistant", "content": "Here's the CPU usage plot!"})
        # Display CPU usage plot
        st.plotly_chart(generate_usage_plot(cpu_points, {}))
    elif "memory" in user_input.lower():
        st.session_state.messages.append({"role": "assistant", "content": "Here's the Memory usage plot!"})
        # Display Memory usage plot
        st.plotly_chart(generate_usage_plot({}, memory_points))
    else:
        messages.append({"role": "user", "content": format_input_query(logs, cpu_data, memory_data, user_input)})
        response = client.chat.completions.create(
                model=model,
                messages=messages
        )
        st.session_state.messages.append({"role": "assistant", "content":  response.choices[0].message.content})
        st.markdown( response.choices[0].message.content)

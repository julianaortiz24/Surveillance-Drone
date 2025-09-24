import tkinter as tk
import json
from tkinter import Canvas

ROUTE_FILE = "route.json"

def launch_route_planner():
    def save_route():
        with open(ROUTE_FILE, "w") as f:
            json.dump(route_points, f)
        root.destroy()

    def on_click(event):
        x, y = event.x, event.y
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="blue")
        if route_points:
            last_x, last_y = route_points[-1]
            canvas.create_line(last_x, last_y, x, y, fill="blue", width=2)
        route_points.append((x, y))

    root = tk.Tk()
    root.title("Route Planner")
    root.attributes('-fullscreen', True)  # Fullscreen window

    canvas = Canvas(root, bg="white")
    canvas.pack(fill="both", expand=True)

    route_points = []

    canvas.bind("<Button-1>", on_click)

    done_button = tk.Button(root, text="Done", font=("Segoe UI", 12), bg="green", fg="white", command=save_route)
    done_button.place(relx=0.5, rely=0.95, anchor="center")

    root.mainloop()

if __name__ == "__main__":
    launch_route_planner()

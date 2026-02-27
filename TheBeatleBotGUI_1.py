import customtkinter as ctk
from functools import partial
from TheBeatleBot_1 import (
    model_paul, model_ringo, model_john,
    predict_george, predict_paul, predict_ringo, predict_john
)

class ChordPredictorGUI:
    def __init__(self, root):
        self.chord_history = []

        root.title("Bayesian Chord Predictor")
        root.geometry("1600x800")
        ctk.set_appearance_mode("dark")

        # Rocket palette colors
        self.bg_color = "#0f0f0f"
        self.chord_color = "#1f1f2e"
        self.suggestion_color = "#2e8b57"
        self.text_color = "#f5f5f5"

        # Chord and color options
        romans = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
        chords = romans + [f"b{romans[i]}" for i in range(1, 7)] + [f"{romans[i]}#" for i in range(1, 7)]
        colors = ['maj', 'min', 'aug', 'dim', 'min7', 'maj7', '7', 'sus2', 'sus4']

        # Input area
        input_frame = ctk.CTkFrame(root, fg_color=self.bg_color)
        input_frame.pack(pady=10)

        ctk.CTkLabel(input_frame, text="Chord 1 (prev2):", text_color=self.text_color).grid(row=0, column=0)
        self.chord1_entry = ctk.CTkOptionMenu(input_frame, values=chords)
        self.chord1_entry.grid(row=0, column=1, padx=10)

        ctk.CTkLabel(input_frame, text="Color 1:", text_color=self.text_color).grid(row=0, column=2)
        self.color1_entry = ctk.CTkOptionMenu(input_frame, values=colors)
        self.color1_entry.grid(row=0, column=3, padx=10)

        ctk.CTkLabel(input_frame, text="Chord 2 (prev1):", text_color=self.text_color).grid(row=1, column=0)
        self.chord2_entry = ctk.CTkOptionMenu(input_frame, values=chords)
        self.chord2_entry.grid(row=1, column=1, padx=10)

        ctk.CTkLabel(input_frame, text="Color 2:", text_color=self.text_color).grid(row=1, column=2)
        self.color2_entry = ctk.CTkOptionMenu(input_frame, values=colors)
        self.color2_entry.grid(row=1, column=3, padx=10)

        # Predictor selection
        ctk.CTkLabel(input_frame, text="Predictor:", text_color=self.text_color).grid(row=2, column=0)
        self.predictor_choice = ctk.CTkOptionMenu(input_frame, values=[
            "predict_george", "predict_paul", "predict_john", "predict_ringo"
        ])
        self.predictor_choice.grid(row=2, column=1, padx=10)

        self.predict_button = ctk.CTkButton(input_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=2, pady=10)

        self.reset_button = ctk.CTkButton(input_frame, text="Reset", fg_color="#8b0000", command=self.reset)
        self.reset_button.grid(row=2, column=3, pady=10)

        # Canvas for history and suggestions
        self.canvas_frame = ctk.CTkScrollableFrame(root, width=1500, height=650, orientation="horizontal")
        self.canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.canvas_frame._parent_canvas.xview_moveto(1.0)

        self.buttons = []

    def predict(self):
        p2 = self.chord1_entry.get()
        c2 = self.color1_entry.get()
        p1 = self.chord2_entry.get()
        c1 = self.color2_entry.get()
        predictor_name = self.predictor_choice.get()

        # Record input
        self.chord_history.append((p2, c2))
        self.chord_history.append((p1, c1))

        # Choose correct model and predictor
        if predictor_name == "predict_george":
            results = predict_george(model_paul, p2, c2, p1, c1)
        elif predictor_name == "predict_paul":
            results = predict_paul(model_paul, p2, c2, p1, c1)
        elif predictor_name == "predict_john":
            results = predict_john(model_john, p2, c2, p1, c1)
        elif predictor_name == "predict_ringo":
            results = predict_ringo(model_ringo, p2, c2, p1, c1)
        else:
            results = {}

        self.draw_tree(results)

    def draw_tree(self, results):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        spacing_x = 170

        for i, (chord, color) in enumerate(self.chord_history):
            label = f"{chord}\n({color})"
            btn = ctk.CTkButton(
                master=self.canvas_frame,
                text=label,
                width=120,
                height=60,
                fg_color=self.chord_color,
                text_color=self.text_color,
                font=("Helvetica", 14, "bold"),
                corner_radius=12
            )
            btn.grid(row=0, column=i, padx=spacing_x // 4, pady=10)

        suggestions = list(results.keys())[:4]
        for i, label in enumerate(suggestions):
            btn = ctk.CTkButton(
                master=self.canvas_frame,
                text=label,
                width=120,
                height=60,
                fg_color=self.suggestion_color,
                text_color=self.text_color,
                font=("Helvetica", 14, "bold"),
                corner_radius=12,
                command=partial(self.choose_suggestion, label)
            )
            btn.grid(row=1 + i, column=len(self.chord_history) - 1)

        self.canvas_frame._parent_canvas.update_idletasks()
        self.canvas_frame._parent_canvas.xview_moveto(1.0)

    def choose_suggestion(self, label):
        try:
            degree, color = label.split(" (")
            color = color.rstrip(")")
        except:
            degree, color = label, "unknown"

        self.chord_history.append((degree, color))
        last_two = self.chord_history[-2:]
        predictor_name = self.predictor_choice.get()

        if predictor_name == "predict_george":
            results = predict_george(model_paul, last_two[0][0], last_two[0][1], last_two[1][0], last_two[1][1])
        elif predictor_name == "predict_paul":
            results = predict_paul(model_paul, last_two[0][0], last_two[0][1], last_two[1][0], last_two[1][1])
        elif predictor_name == "predict_john":
            results = predict_john(model_john, last_two[0][0], last_two[0][1], last_two[1][0], last_two[1][1])
        elif predictor_name == "predict_ringo":
            results = predict_ringo(model_ringo, last_two[0][0], last_two[0][1], last_two[1][0], last_two[1][1])
        else:
            results = {}

        self.draw_tree(results)

    def reset(self):
        self.chord_history = []
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ChordPredictorGUI(root)
    root.mainloop()

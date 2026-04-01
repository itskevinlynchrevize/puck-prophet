"""
NHL Dashboard — Flask App
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import os
import json
import threading

from engine import NHLModel, fetch_schedule, fetch_game_stats, API_BASE

app = Flask(__name__)

# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

model = NHLModel()
predictions_cache = {}  # date -> list of predictions
results_cache = {}      # date -> list of results
accuracy_log = []       # list of {date, correct, total}

DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_state_path():
    return os.path.join(DATA_DIR, "model_state.json")

def get_predictions_path():
    return os.path.join(DATA_DIR, "predictions.json")

def get_accuracy_path():
    return os.path.join(DATA_DIR, "accuracy.json")


def save_predictions():
    with open(get_predictions_path(), "w") as f:
        json.dump(predictions_cache, f, indent=2, default=str)

def load_predictions():
    global predictions_cache
    path = get_predictions_path()
    if os.path.exists(path):
        with open(path) as f:
            predictions_cache = json.load(f)

def save_accuracy():
    with open(get_accuracy_path(), "w") as f:
        json.dump(accuracy_log, f, indent=2)

def load_accuracy():
    global accuracy_log
    path = get_accuracy_path()
    if os.path.exists(path):
        with open(path) as f:
            accuracy_log = json.load(f)


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_model():
    """Train model on available season data."""
    print("  Initializing model...")

    season_files = [
        ("2024-25", os.path.join(DATA_DIR, "nhl_2024_25_games.csv"), 0.030),
        ("2025-26", os.path.join(DATA_DIR, "nhl_2025_26_games.csv"), 0.015),
    ]

    for label, filepath, lr in season_files:
        if os.path.exists(filepath):
            print(f"  Training on {label}...")
            result = model.train_on_season(filepath, base_lr=lr, reset=True)
            print(f"    {result['accuracy']:.1f}% ({result['predictions']} predictions)")

    model.save_state(get_state_path())
    load_predictions()
    load_accuracy()
    print("  Model ready.")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/predictions/today")
def api_today():
    """Get predictions for today's games."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Check cache first
    if today in predictions_cache:
        return jsonify({"date": today, "predictions": predictions_cache[today]})

    # Fetch today's schedule
    games = fetch_schedule(today)
    upcoming = [g for g in games if g["state"] not in ["FINAL", "OFF"]]

    if not upcoming:
        return jsonify({"date": today, "predictions": [], "message": "No games scheduled"})

    predictions = []
    for game in upcoming:
        home = game["home_team"]; away = game["away_team"]
        home_rest = model.get_rest_days(home)
        away_rest = model.get_rest_days(away)

        # Get likely starters (would be overridden by manual input)
        home_gid = None; away_gid = None

        pred = model.predict(home, away, home_rest, away_rest, home_gid, away_gid)
        if pred:
            pred["game_id"] = game["game_id"]
            pred["start_time"] = game.get("start_time", "")
            predictions.append(pred)

    # Sort by confidence descending
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    predictions_cache[today] = predictions
    save_predictions()

    return jsonify({"date": today, "predictions": predictions})


@app.route("/api/predictions/<date_str>")
def api_predictions_by_date(date_str):
    """Get predictions for a specific date."""
    if date_str in predictions_cache:
        return jsonify({"date": date_str, "predictions": predictions_cache[date_str]})
    return jsonify({"date": date_str, "predictions": [], "message": "No predictions for this date"})


@app.route("/api/results/yesterday")
def api_yesterday():
    """Get yesterday's results and check predictions."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch yesterday's completed games
    games = fetch_schedule(yesterday)
    completed = [g for g in games if g["state"] in ["FINAL", "OFF"]]

    if not completed:
        return jsonify({"date": yesterday, "results": [], "message": "No completed games"})

    # Check against our predictions
    yesterday_preds = predictions_cache.get(yesterday, [])
    pred_map = {p["game_id"]: p for p in yesterday_preds if "game_id" in p}

    results = []
    day_correct = 0
    day_total = 0

    for game in completed:
        gid = game["game_id"]
        home = game["home_team"]; away = game["away_team"]
        home_score = game.get("home_score"); away_score = game.get("away_score")

        if home_score is None or away_score is None:
            continue

        actual_winner = home if home_score > away_score else away
        margin = abs(home_score - away_score)

        result = {
            "game_id": gid,
            "home": home, "away": away,
            "home_score": home_score, "away_score": away_score,
            "actual_winner": actual_winner,
            "margin": margin,
        }

        if gid in pred_map:
            pred = pred_map[gid]
            correct = pred["pick"] == actual_winner
            result["predicted"] = True
            result["pick"] = pred["pick"]
            result["confidence"] = pred["confidence"]
            result["tier"] = pred["tier"]
            result["correct"] = correct
            result["puck_line"] = pred.get("puck_line")
            result["puck_line_correct"] = (
                margin > 1.5 and pred.get("puck_line") == actual_winner
            ) if pred.get("puck_line") else None

            day_total += 1
            if correct: day_correct += 1
        else:
            result["predicted"] = False

        results.append(result)

    # Log accuracy
    if day_total > 0:
        accuracy_log.append({
            "date": yesterday,
            "correct": day_correct,
            "total": day_total,
            "accuracy": round(day_correct/day_total*100, 1),
        })
        save_accuracy()

    return jsonify({
        "date": yesterday,
        "results": results,
        "summary": {
            "correct": day_correct,
            "total": day_total,
            "accuracy": round(day_correct/day_total*100, 1) if day_total > 0 else None,
        }
    })


@app.route("/api/accuracy")
def api_accuracy():
    """Get rolling accuracy stats."""
    total_correct = sum(d["correct"] for d in accuracy_log)
    total_preds = sum(d["total"] for d in accuracy_log)
    overall = round(total_correct/total_preds*100, 1) if total_preds > 0 else 0

    # Last 7 days
    recent = accuracy_log[-7:] if accuracy_log else []
    recent_correct = sum(d["correct"] for d in recent)
    recent_total = sum(d["total"] for d in recent)
    recent_acc = round(recent_correct/recent_total*100, 1) if recent_total > 0 else 0

    return jsonify({
        "overall": {"correct": total_correct, "total": total_preds, "accuracy": overall},
        "recent_7d": {"correct": recent_correct, "total": recent_total, "accuracy": recent_acc},
        "daily_log": accuracy_log[-30:],
    })


@app.route("/api/weights")
def api_weights():
    """Get current model weights."""
    return jsonify({"weights": model.get_weights()})


@app.route("/api/goalie/override", methods=["POST"])
def api_goalie_override():
    """Manually set a starting goalie and recalculate prediction."""
    data = request.json
    # TODO: implement goalie override recalculation
    return jsonify({"status": "ok", "message": "Goalie override not yet implemented"})


@app.route("/api/parlay", methods=["POST"])
def api_parlay():
    """Calculate parlay probability from selected games."""
    data = request.json
    game_ids = data.get("game_ids", [])

    today = datetime.now().strftime("%Y-%m-%d")
    today_preds = predictions_cache.get(today, [])
    # Build pred_map with both string and int keys to handle type mismatches
    pred_map = {}
    for p in today_preds:
        if "game_id" in p:
            pred_map[str(p["game_id"])] = p
            pred_map[p["game_id"]] = p

    selected = []
    combined_prob = 1.0

    for gid in game_ids:
        # Try both string and original type
        pred = pred_map.get(str(gid)) or pred_map.get(gid)
        if pred:
            prob = pred["raw_confidence"]
            # Use the confidence in the predicted direction
            if pred["pick"] != pred["home"]:
                prob = 1 - prob
            actual_prob = max(prob, 1 - prob)
            combined_prob *= actual_prob
            selected.append({
                "game_id": str(gid),
                "pick": pred["pick"],
                "confidence": pred["confidence"],
                "home": pred["home"],
                "away": pred["away"],
            })

    # Convert to American odds
    if combined_prob > 0.5:
        american_odds = round(-100 * combined_prob / (1 - combined_prob))
    else:
        american_odds = round(100 * (1 - combined_prob) / combined_prob)

    return jsonify({
        "legs": selected,
        "combined_probability": round(combined_prob * 100, 1),
        "fair_american_odds": american_odds,
        "leg_count": len(selected),
    })


# ============================================================
# STARTUP
# ============================================================

# Initialize model on import (runs for both gunicorn and direct execution)
initialize_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

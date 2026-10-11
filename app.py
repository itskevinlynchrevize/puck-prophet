"""
NHL Dashboard — Flask App
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import os
import json
import time
import threading
import atexit
import logging

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except ImportError:
    EASTERN = None

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from engine import NHLModel, fetch_schedule, fetch_game_stats, fetch_probable_goalies, API_BASE

app = Flask(__name__)

# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

model = NHLModel()
predictions_cache = {}  # date -> list of predictions
results_cache = {}      # date -> list of results
accuracy_log = []       # list of {date, correct, total}
season_accuracy = {}    # season_label -> {correct, total, accuracy}

DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# TIME HELPERS
# ============================================================

def eastern_now():
    """Get current datetime in Eastern time."""
    if EASTERN:
        return datetime.now(EASTERN)
    return datetime.now()

def eastern_today():
    """Get today's date string in Eastern time."""
    return eastern_now().strftime("%Y-%m-%d")

def eastern_yesterday():
    """Get yesterday's date string in Eastern time."""
    return (eastern_now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ============================================================
# PERSISTENCE HELPERS
# ============================================================

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
    """Train model on available season data, then ingest current season."""
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
            season_accuracy[label] = {
                "correct": result["correct"],
                "total": result["predictions"],
                "accuracy": result["accuracy"],
            }

    # Reset team/goalie stats so current season starts clean.
    # Weights are preserved — only the per-team accumulators are wiped.
    all_teams = list(model.team_stats.keys())
    if all_teams:
        print("  Resetting team stats for new season...")
        model.reset_season_stats(all_teams)

    # Pull completed 2026-27 games from the NHL API
    # This rebuilds team stats, goalie names, rest days from scratch
    try:
        games_ingested = model.ingest_current_season(base_lr=0.012, data_dir=DATA_DIR)
        if games_ingested > 0:
            print(f"  Current season: {games_ingested} games ingested.")
    except Exception as e:
        print(f"  Warning: Could not ingest current season: {e}")

    model.save_state(get_state_path())
    load_accuracy()

    # Clear today's cached predictions so they regenerate with fresh model data
    today = eastern_today()
    if today in predictions_cache:
        del predictions_cache[today]
        save_predictions()
    else:
        load_predictions()
        if today in predictions_cache:
            del predictions_cache[today]
            save_predictions()

    print("  Model ready.")


# ============================================================
# DAILY CRON JOB
# ============================================================

def daily_refresh():
    """Daily job: ingest new games, score yesterday's predictions, refresh cache.

    Runs at 10am ET via APScheduler. Steps:
      1. Fetch yesterday's completed games from the NHL API.
      2. For each game not already ingested, pull stats and run through
         train_on_game() so team/goalie stats stay current.
      3. Score yesterday's predictions against actual results (accuracy log).
      4. Clear today's prediction cache so fresh predictions generate.
      5. Save model state and ingestion cache.
    """
    print(f"[CRON] daily_refresh starting at {eastern_now()}")

    yesterday = eastern_yesterday()

    # ---- 1. Ingest yesterday's games ---------------------
    games = fetch_schedule(yesterday)
    completed = [g for g in games if g["state"] in ["FINAL", "OFF"]]
    print(f"[CRON] {len(completed)} completed games found for {yesterday}")

    # Load ingestion cache
    cache_path = os.path.join(DATA_DIR, "ingested_games.json")
    ingested_ids = set()
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                ingested_ids = set(json.load(f))
    except Exception:
        pass

    import pandas as pd

    new_ingested = 0
    for game in completed:
        gid = game["game_id"]
        if str(gid) in ingested_ids or gid in ingested_ids:
            continue

        try:
            time.sleep(0.8)
            stats = fetch_game_stats(gid)
            if stats:
                stats["home_team"] = game["home_team"]
                stats["away_team"] = game["away_team"]
                stats["date"] = pd.Timestamp(yesterday)
                stats["home_rest"] = model.get_rest_days(game["home_team"])
                stats["away_rest"] = model.get_rest_days(game["away_team"])

                model.train_on_game(stats, base_lr=0.012)
                ingested_ids.add(str(gid))
                new_ingested += 1
        except Exception as e:
            print(f"[CRON] Error ingesting game {gid}: {e}")

    # Save updated ingestion cache
    if new_ingested > 0:
        try:
            with open(cache_path, "w") as f:
                json.dump(sorted(ingested_ids), f)
        except Exception as e:
            print(f"[CRON] Warning: could not save ingestion cache: {e}")

    print(f"[CRON] Ingested {new_ingested} new games")

    # ---- 2. Score yesterday's predictions ----------------
    already_logged = any(d["date"] == yesterday for d in accuracy_log)

    if not already_logged and completed:
        yesterday_preds = predictions_cache.get(yesterday, [])
        pred_map = {p["game_id"]: p for p in yesterday_preds if "game_id" in p}

        day_correct = 0
        day_total = 0
        for game in completed:
            gid = game["game_id"]
            if gid not in pred_map:
                continue
            home_score = game.get("home_score")
            away_score = game.get("away_score")
            if home_score is None or away_score is None:
                continue

            actual_winner = game["home_team"] if home_score > away_score else game["away_team"]
            pred = pred_map[gid]
            if pred["pick"] == actual_winner:
                day_correct += 1
            day_total += 1

        if day_total > 0:
            accuracy_log.append({
                "date": yesterday,
                "correct": day_correct,
                "total": day_total,
                "accuracy": round(day_correct / day_total * 100, 1),
            })
            save_accuracy()
            print(f"[CRON] Accuracy logged for {yesterday}: {day_correct}/{day_total}")
        else:
            print(f"[CRON] No predictions to score for {yesterday}")
    elif already_logged:
        print(f"[CRON] Accuracy for {yesterday} already logged, skipping")

    # ---- 3. Clear today's cached predictions -------------
    today = eastern_today()
    if today in predictions_cache:
        del predictions_cache[today]
        save_predictions()
        print(f"[CRON] Cleared prediction cache for {today}")

    # ---- 4. Save model state -----------------------------
    model.save_state(get_state_path())
    print(f"[CRON] daily_refresh complete at {eastern_now()}")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/predictions/today")
def api_today():
    """Get predictions for today's games."""
    today = eastern_today()

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

        # Try to get probable starters from the API
        home_gid = None; away_gid = None
        try:
            goalies = fetch_probable_goalies(game["game_id"])
            if goalies:
                home_gid = goalies.get("home_goalie_id")
                away_gid = goalies.get("away_goalie_id")
                # Store names so compute_features can find them
                if home_gid and goalies.get("home_goalie_name"):
                    model.goalie_names[home_gid] = goalies["home_goalie_name"]
                if away_gid and goalies.get("away_goalie_name"):
                    model.goalie_names[away_gid] = goalies["away_goalie_name"]
        except:
            pass

        # Fall back to most recent starter for each team
        if not home_gid:
            home_gid = model.get_likely_starter(home)
        if not away_gid:
            away_gid = model.get_likely_starter(away)

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


@app.route("/api/results/today")
def api_today_results():
    """Get today's completed game results (no accuracy logging)."""
    today = eastern_today()

    games = fetch_schedule(today)
    completed = [g for g in games if g["state"] in ["FINAL", "OFF"]]

    if not completed:
        return jsonify({"date": today, "results": [], "message": "No completed games yet today"})

    today_preds = predictions_cache.get(today, [])
    pred_map = {p["game_id"]: p for p in today_preds if "game_id" in p}

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

            day_total += 1
            if correct: day_correct += 1
        else:
            result["predicted"] = False

        results.append(result)

    return jsonify({
        "date": today,
        "results": results,
        "summary": {
            "correct": day_correct,
            "total": day_total,
            "accuracy": round(day_correct/day_total*100, 1) if day_total > 0 else None,
        }
    })


@app.route("/api/results/yesterday")
def api_yesterday():
    """Get yesterday's results and check predictions."""
    yesterday = eastern_yesterday()

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

    # Log accuracy (skip if already logged by cron or a previous request)
    already_logged = any(d["date"] == yesterday for d in accuracy_log)
    if day_total > 0 and not already_logged:
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
    # Live prediction accuracy (current season daily tracking)
    live_correct = sum(d["correct"] for d in accuracy_log)
    live_total = sum(d["total"] for d in accuracy_log)

    # All-time: training seasons + live predictions
    all_correct = sum(s["correct"] for s in season_accuracy.values()) + live_correct
    all_total = sum(s["total"] for s in season_accuracy.values()) + live_total
    all_acc = round(all_correct/all_total*100, 1) if all_total > 0 else 0

    # Last season (2025-26)
    last_season = season_accuracy.get("2025-26", {"correct": 0, "total": 0, "accuracy": 0})

    # Last 7 days of live predictions
    recent = accuracy_log[-7:] if accuracy_log else []
    recent_correct = sum(d["correct"] for d in recent)
    recent_total = sum(d["total"] for d in recent)
    recent_acc = round(recent_correct/recent_total*100, 1) if recent_total > 0 else 0

    return jsonify({
        "all_time": {"correct": all_correct, "total": all_total, "accuracy": all_acc},
        "last_season": last_season,
        "recent_7d": {"correct": recent_correct, "total": recent_total, "accuracy": recent_acc},
        "daily_log": accuracy_log[-30:],
        "seasons": season_accuracy,
    })


@app.route("/api/weights")
def api_weights():
    """Get current model weights."""
    return jsonify({"weights": model.get_weights()})


@app.route("/api/refresh", methods=["POST"])
def api_manual_refresh():
    """Manually trigger the daily refresh (same as the cron job)."""
    try:
        daily_refresh()
        return jsonify({"status": "ok", "message": "Refresh complete"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/goalies/<team>")
def api_goalies_for_team(team):
    """List known goalies for a team with their stats."""
    team = team.upper()
    goalie_ids = model.team_goalies.get(team, set())

    goalies = []
    for gid in goalie_ids:
        gs = model.goalie_stats.get(gid, {})
        games = gs.get("games", 0)
        if games == 0:
            continue
        sa = gs.get("shots_against", 0)
        sv_pct = gs.get("saves", 0) / sa if sa > 0 else 0
        goalies.append({
            "goalie_id": gid,
            "name": model.goalie_names.get(gid, "Unknown"),
            "games": games,
            "wins": gs.get("wins", 0),
            "sv_pct": round(sv_pct, 3),
            "goals_against": gs.get("goals_against", 0),
            "is_starter": model.team_last_starter.get(team) == gid,
        })

    # Sort: most recent starter first, then by games played
    goalies.sort(key=lambda g: (-g["is_starter"], -g["games"]))
    return jsonify({"team": team, "goalies": goalies})


@app.route("/api/goalie/override", methods=["POST"])
def api_goalie_override():
    """Manually set a starting goalie and recalculate prediction."""
    data = request.json
    game_id = data.get("game_id")
    side = data.get("side")          # "home" or "away"
    goalie_id = data.get("goalie_id")

    if not game_id or not side or goalie_id is None:
        return jsonify({"status": "error", "message": "game_id, side, and goalie_id required"}), 400

    today = eastern_today()
    today_preds = predictions_cache.get(today, [])

    # Find the prediction for this game
    pred_idx = None
    for i, p in enumerate(today_preds):
        if str(p.get("game_id")) == str(game_id):
            pred_idx = i
            break

    if pred_idx is None:
        return jsonify({"status": "error", "message": "Game not found in today's predictions"}), 404

    old_pred = today_preds[pred_idx]
    home = old_pred["home"]
    away = old_pred["away"]

    # Determine goalie IDs for recalculation
    if side == "home":
        home_gid = int(goalie_id)
        away_gid = None
        # Try to find the away goalie from the original prediction
        for gid, name in model.goalie_names.items():
            if name == old_pred.get("away_goalie_name"):
                away_gid = gid
                break
        if away_gid is None:
            away_gid = model.get_likely_starter(away)
    else:
        away_gid = int(goalie_id)
        home_gid = None
        for gid, name in model.goalie_names.items():
            if name == old_pred.get("home_goalie_name"):
                home_gid = gid
                break
        if home_gid is None:
            home_gid = model.get_likely_starter(home)

    # Recalculate
    new_pred = model.predict(
        home, away,
        old_pred["home_rest"], old_pred["away_rest"],
        home_gid, away_gid
    )

    if not new_pred:
        return jsonify({"status": "error", "message": "Could not recalculate prediction"}), 500

    # Preserve metadata from original
    new_pred["game_id"] = old_pred["game_id"]
    new_pred["start_time"] = old_pred.get("start_time", "")

    # Mark which side was overridden
    if side == "home":
        new_pred["home_goalie_override"] = True
    else:
        new_pred["away_goalie_override"] = True

    # Update cache
    today_preds[pred_idx] = new_pred
    predictions_cache[today] = today_preds
    save_predictions()

    return jsonify({"status": "ok", "prediction": new_pred})


@app.route("/api/parlay", methods=["POST"])
def api_parlay():
    """Calculate parlay probability from selected games."""
    data = request.json
    game_ids = data.get("game_ids", [])

    today = eastern_today()
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

# Start daily scheduler
# Guard: skip if running inside Flask's reloader child process
if os.environ.get("WERKZEUG_RUN_MAIN") != "true" or not app.debug:
    scheduler = BackgroundScheduler(daemon=True)

    # 10:00 AM Eastern every day
    if EASTERN:
        trigger = CronTrigger(hour=10, minute=0, timezone=EASTERN)
    else:
        # Fallback: UTC 14:00 ≈ ET 10:00 (EDT) / 09:00 (EST)
        trigger = CronTrigger(hour=14, minute=0)

    scheduler.add_job(daily_refresh, trigger, id="daily_refresh",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.start()
    print(f"  Scheduler started — daily_refresh at 10:00 AM ET")

    atexit.register(lambda: scheduler.shutdown(wait=False))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

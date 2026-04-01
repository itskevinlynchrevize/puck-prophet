"""
NHL Prediction Engine
Core model logic packaged for the dashboard.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

API_BASE = "https://api-web.nhle.com/v1"

DIVISIONS = {
    "atlantic": ["FLA", "TOR", "BOS", "TBL", "BUF", "OTT", "MTL", "DET"],
    "metro": ["NYR", "CAR", "PHI", "NYI", "WSH", "PIT", "NJD", "CBJ"],
    "central": ["DAL", "WPG", "COL", "NSH", "MIN", "STL", "UTA", "ARI", "CHI"],
    "pacific": ["VAN", "EDM", "VGK", "LAK", "SEA", "CGY", "ANA", "SJS"],
}
TEAM_TO_DIV = {}
for div, teams in DIVISIONS.items():
    for t in teams:
        TEAM_TO_DIV[t] = div

def same_division(t1, t2):
    d1 = TEAM_TO_DIV.get(t1); d2 = TEAM_TO_DIV.get(t2)
    return d1 == d2 if d1 and d2 else False

def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def fresh_team_stats():
    return {"games": 0, "gf": 0, "ga": 0, "pp_goals": 0, "pp_opps": 0,
            "pk_opps": 0, "pk_ga": 0, "giveaways": 0, "pim": 0,
            "saves": 0, "shots_against": 0, "wins": 0, "losses": 0,
            "points": 0, "ot_games": 0}

def fresh_goalie_stats():
    return {"games": 0, "wins": 0, "saves": 0, "shots_against": 0, "goals_against": 0}


# ============================================================
# NHL API HELPERS
# ============================================================

def fetch_schedule(date_str):
    """Fetch games for a given date."""
    try:
        resp = requests.get(f"{API_BASE}/schedule/{date_str}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for week in data.get("gameWeek", []):
            if week.get("date") == date_str:
                for game in week.get("games", []):
                    if game.get("gameType") == 2:
                        games.append({
                            "game_id": game.get("id"),
                            "home_team": game.get("homeTeam", {}).get("abbrev", ""),
                            "away_team": game.get("awayTeam", {}).get("abbrev", ""),
                            "state": game.get("gameState", ""),
                            "start_time": game.get("startTimeUTC", ""),
                            "home_score": game.get("homeTeam", {}).get("score"),
                            "away_score": game.get("awayTeam", {}).get("score"),
                        })
        return games
    except Exception as e:
        print(f"  Error fetching schedule for {date_str}: {e}")
        return []


def fetch_game_stats(game_id):
    """Pull boxscore + right-rail + PBP for one game."""
    box = rr = pbp = None
    try: box = requests.get(f"{API_BASE}/gamecenter/{game_id}/boxscore", timeout=15).json()
    except: pass
    try: rr = requests.get(f"{API_BASE}/gamecenter/{game_id}/right-rail", timeout=15).json()
    except: pass
    try: pbp = requests.get(f"{API_BASE}/gamecenter/{game_id}/play-by-play", timeout=15).json()
    except: pass

    if not box: return None

    try:
        home_goals = box.get("homeTeam", {}).get("score", 0)
        away_goals = box.get("awayTeam", {}).get("score", 0)
        outcome = box.get("gameOutcome", {})
        lpt = outcome.get("lastPeriodType", "REG")

        h_sog=0;a_sog=0;h_fo=50.0;a_fo=50.0;h_ppg=0;h_ppo=0;a_ppg=0;a_ppo=0
        h_pim=0;a_pim=0;h_give=0;a_give=0

        if rr:
            for s in rr.get("teamGameStats", []):
                cat=s.get("category",""); hv=s.get("homeValue",""); av=s.get("awayValue","")
                if cat=="sog": h_sog=int(hv) if hv else 0; a_sog=int(av) if av else 0
                elif cat=="faceoffWinningPctg":
                    try: h_fo=float(str(hv).replace("%","")); a_fo=float(str(av).replace("%",""))
                    except: pass
                elif cat=="powerPlay":
                    try: p=str(hv).split("/"); h_ppg=int(p[0]); h_ppo=int(p[1])
                    except: pass
                    try: p=str(av).split("/"); a_ppg=int(p[0]); a_ppo=int(p[1])
                    except: pass
                elif cat=="pim": h_pim=int(hv) if hv else 0; a_pim=int(av) if av else 0
                elif cat=="giveaways": h_give=int(hv) if hv else 0; a_give=int(av) if av else 0

        h_starter_id=None;h_starter_name="";a_starter_id=None;a_starter_name=""
        h_en=0;a_en=0

        if pbp:
            home_tid=pbp.get("homeTeam",{}).get("id")
            away_tid=pbp.get("awayTeam",{}).get("id")
            pmap={}
            for p in pbp.get("rosterSpots",[]):
                pid=p.get("playerId")
                if pid: pmap[pid]=f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}".strip()
            hgs=[]; ags=[]
            for play in pbp.get("plays",[]):
                per=play.get("periodDescriptor",{}).get("number",0)
                if per>1: break
                et=play.get("typeDescKey","")
                if et in ["shot-on-goal","goal","save"]:
                    d=play.get("details",{})
                    gid_p=d.get("goalieInNetId"); oid=d.get("eventOwnerTeamId")
                    if gid_p:
                        if oid==home_tid and not ags: ags.append(gid_p)
                        elif oid==away_tid and not hgs: hgs.append(gid_p)
            if hgs: h_starter_id=hgs[0]; h_starter_name=pmap.get(h_starter_id,"")
            if ags: a_starter_id=ags[0]; a_starter_name=pmap.get(a_starter_id,"")
            for play in pbp.get("plays",[]):
                et=play.get("typeDescKey","")
                pn=play.get("periodDescriptor",{}).get("number",0)
                d=play.get("details",{})
                if et=="goal" and pn<=3:
                    stid=d.get("eventOwnerTeamId"); gin=d.get("goalieInNetId")
                    if gin is None or gin==0:
                        if stid==home_tid: h_en+=1
                        elif stid==away_tid: a_en+=1

        return {
            "game_id": game_id, "home_goals": home_goals, "away_goals": away_goals,
            "home_win": 1 if home_goals > away_goals else 0,
            "home_pp_goals": h_ppg, "home_pp_opps": h_ppo,
            "away_pp_goals": a_ppg, "away_pp_opps": a_ppo,
            "home_pk_opps": a_ppo, "away_pk_opps": h_ppo,
            "home_giveaways": h_give, "away_giveaways": a_give,
            "home_pim": h_pim, "away_pim": a_pim,
            "home_sog": h_sog, "away_sog": a_sog,
            "went_to_ot": 1 if lpt in ["OT","SO"] else 0,
            "went_to_so": 1 if lpt=="SO" else 0,
            "home_starter_id": h_starter_id, "home_starter_name": h_starter_name,
            "away_starter_id": a_starter_id, "away_starter_name": a_starter_name,
            "home_empty_net_goals": h_en, "away_empty_net_goals": a_en,
        }
    except:
        return None


# ============================================================
# MODEL
# ============================================================

class NHLModel:
    """The prediction engine."""

    def __init__(self):
        self.weights = {
            "goal_diff": 0.05, "rest_diff": 0.02, "give_diff": 0.02,
            "div_flag": -0.01, "b2b_flag": 0.02, "pk_burden": 0.02,
            "goalie_delta": 0.02, "pp_diff": 0.01, "bias": 0.05,
        }
        self.lr_mult = {
            "goal_diff": 1.0, "rest_diff": 2.0, "give_diff": 2.5,
            "div_flag": 2.0, "b2b_flag": 2.0, "pk_burden": 2.5,
            "goalie_delta": 3.0, "pp_diff": 2.0, "bias": 1.0,
        }
        self.team_stats = {}
        self.goalie_stats = defaultdict(fresh_goalie_stats)
        self.goalie_names = {}
        self.last_game_date = {}
        self.reg_lambda = 0.002
        self.min_games = 10

    def reset_season_stats(self, teams):
        """Reset team and goalie stats for a new season."""
        for t in teams:
            self.team_stats[t] = fresh_team_stats()
        self.goalie_stats = defaultdict(fresh_goalie_stats)
        self.last_game_date = {}

    def ensure_team(self, team):
        if team not in self.team_stats:
            self.team_stats[team] = fresh_team_stats()

    def compute_features(self, home, away, home_rest, away_rest,
                         home_gid=None, away_gid=None):
        """Compute features for a matchup."""
        h = self.team_stats.get(home)
        a = self.team_stats.get(away)
        if not h or not a or h["games"] < self.min_games or a["games"] < self.min_games:
            return None

        hg = h["games"]; ag = a["games"]
        h_gf = h["gf"]/hg; h_ga = h["ga"]/hg
        a_gf = a["gf"]/ag; a_ga = a["ga"]/ag
        h_pp = (h["pp_goals"]/h["pp_opps"]*100) if h["pp_opps"]>0 else 20
        a_pp = (a["pp_goals"]/a["pp_opps"]*100) if a["pp_opps"]>0 else 20
        h_give = h["giveaways"]/hg; a_give = a["giveaways"]/ag
        h_pk_pg = h["pk_opps"]/hg; a_pk_pg = a["pk_opps"]/ag
        h_team_sv = h["saves"]/h["shots_against"] if h["shots_against"]>0 else 0.9
        a_team_sv = a["saves"]/a["shots_against"] if a["shots_against"]>0 else 0.9

        # Goalie delta
        h_g_delta = 0.0; a_g_delta = 0.0
        hg_sv = h_team_sv; ag_sv = a_team_sv

        if home_gid and self.goalie_stats[home_gid]["games"] >= 5:
            gs = self.goalie_stats[home_gid]
            if gs["shots_against"] > 0:
                starter_sv = gs["saves"]/gs["shots_against"]
                h_g_delta = starter_sv - h_team_sv
                hg_sv = starter_sv

        if away_gid and self.goalie_stats[away_gid]["games"] >= 5:
            gs = self.goalie_stats[away_gid]
            if gs["shots_against"] > 0:
                starter_sv = gs["saves"]/gs["shots_against"]
                a_g_delta = starter_sv - a_team_sv
                ag_sv = starter_sv

        features = {
            "goal_diff": ((h_gf-h_ga)-(a_gf-a_ga))/1.2,
            "rest_diff": (home_rest-away_rest)/2.5,
            "give_diff": (a_give-h_give)/3.0,
            "div_flag": 1.0 if same_division(home, away) else 0.0,
            "b2b_flag": float((-1 if home_rest<=1 else 0)+(1 if away_rest<=1 else 0)),
            "pk_burden": (a_pk_pg-h_pk_pg)/2.0,
            "goalie_delta": (h_g_delta-a_g_delta)/0.015,
            "pp_diff": (h_pp-a_pp)/12.0,
            "bias": 1.0,
        }

        meta = {
            "home_gf": h_gf, "home_ga": h_ga, "away_gf": a_gf, "away_ga": a_ga,
            "home_wins": h["wins"], "home_losses": hg - h["wins"],
            "away_wins": a["wins"], "away_losses": ag - a["wins"],
            "home_win_pct": h["wins"]/hg*100, "away_win_pct": a["wins"]/ag*100,
            "home_pk_pg": h_pk_pg, "away_pk_pg": a_pk_pg,
            "home_goalie_sv": hg_sv, "away_goalie_sv": ag_sv,
            "home_goalie_delta": h_g_delta*100, "away_goalie_delta": a_g_delta*100,
            "home_goalie_name": self.goalie_names.get(home_gid, "Unknown") if home_gid else "Unknown",
            "away_goalie_name": self.goalie_names.get(away_gid, "Unknown") if away_gid else "Unknown",
            "home_goalie_gp": self.goalie_stats[home_gid]["games"] if home_gid else 0,
            "away_goalie_gp": self.goalie_stats[away_gid]["games"] if away_gid else 0,
        }

        return features, meta

    def predict(self, home, away, home_rest, away_rest,
                home_gid=None, away_gid=None):
        """Make a prediction for a single game."""
        result = self.compute_features(home, away, home_rest, away_rest, home_gid, away_gid)
        if result is None:
            return None

        features, meta = result
        z = sum(self.weights[k]*features[k] for k in features if k in self.weights)
        conf = sigmoid(z)
        pred_winner = home if conf >= 0.5 else away
        conf_pct = max(conf, 1-conf) * 100
        conf_margin = abs(conf - 0.5)

        if conf_pct >= 65: tier = "STRONG"
        elif conf_pct >= 58: tier = "LEAN"
        elif conf_pct >= 53: tier = "SLIGHT"
        else: tier = "TOSSUP"

        puck_line = pred_winner if conf_margin > 0.08 else None

        return {
            "home": home, "away": away,
            "pick": pred_winner,
            "confidence": round(conf_pct, 1),
            "raw_confidence": round(float(conf), 4),
            "tier": tier,
            "puck_line": puck_line,
            "divisional": same_division(home, away),
            "home_rest": home_rest, "away_rest": away_rest,
            **meta,
        }

    def train_on_game(self, row, base_lr=0.025):
        """Train on a single completed game."""
        home = row["home_team"]; away = row["away_team"]
        self.ensure_team(home); self.ensure_team(away)
        h = self.team_stats[home]; a = self.team_stats[away]

        home_gid = int(row["home_starter_id"]) if pd.notna(row.get("home_starter_id")) else None
        away_gid = int(row["away_starter_id"]) if pd.notna(row.get("away_starter_id")) else None

        # Store goalie names
        if home_gid and pd.notna(row.get("home_starter_name")) and str(row["home_starter_name"]).strip():
            self.goalie_names[home_gid] = str(row["home_starter_name"]).strip()
        if away_gid and pd.notna(row.get("away_starter_name")) and str(row["away_starter_name"]).strip():
            self.goalie_names[away_gid] = str(row["away_starter_name"]).strip()

        home_rest = row.get("home_rest", 3)
        away_rest = row.get("away_rest", 3)

        predicted = False
        correct = False

        # Predict if enough data
        if h["games"] >= self.min_games and a["games"] >= self.min_games:
            result = self.compute_features(home, away, home_rest, away_rest, home_gid, away_gid)
            if result:
                features, meta = result
                z = sum(self.weights[k]*features[k] for k in features if k in self.weights)
                conf = sigmoid(z)
                pred_home = conf >= 0.5
                actual_home = row["home_win"] == 1
                correct = pred_home == actual_home
                predicted = True

                # Learn
                y = 1.0 if actual_home else 0.0
                error = y - conf
                for k in features:
                    if k in self.weights:
                        self.weights[k] += base_lr * self.lr_mult.get(k,1.0) * (
                            error * features[k] - self.reg_lambda * self.weights[k])

        # Update team stats
        h["games"]+=1; h["gf"]+=row["home_goals"]; h["ga"]+=row["away_goals"]
        h["pp_goals"]+=row["home_pp_goals"]; h["pp_opps"]+=row["home_pp_opps"]
        h["pk_opps"]+=row.get("home_pk_opps",0); h["pk_ga"]+=row["away_pp_goals"]
        h["giveaways"]+=row["home_giveaways"]; h["pim"]+=row["home_pim"]
        h["saves"]+=max(0,row["away_sog"]-row["away_goals"])
        h["shots_against"]+=row["away_sog"]

        a["games"]+=1; a["gf"]+=row["away_goals"]; a["ga"]+=row["home_goals"]
        a["pp_goals"]+=row["away_pp_goals"]; a["pp_opps"]+=row["away_pp_opps"]
        a["pk_opps"]+=row.get("away_pk_opps",0); a["pk_ga"]+=row["home_pp_goals"]
        a["giveaways"]+=row["away_giveaways"]; a["pim"]+=row["away_pim"]
        a["saves"]+=max(0,row["home_sog"]-row["home_goals"])
        a["shots_against"]+=row["home_sog"]

        if row["home_win"]==1:
            h["wins"]+=1; h["points"]+=2
            if row["went_to_ot"]: a["points"]+=1
            else: a["losses"]+=1
        else:
            a["wins"]+=1; a["points"]+=2
            if row["went_to_ot"]: h["points"]+=1
            else: h["losses"]+=1
        if row["went_to_ot"]: h["ot_games"]+=1; a["ot_games"]+=1

        # Goalie stats
        if home_gid:
            gs=self.goalie_stats[home_gid]; gs["games"]+=1
            gs["shots_against"]+=row["away_sog"]; gs["goals_against"]+=row["away_goals"]
            gs["saves"]+=max(0,row["away_sog"]-row["away_goals"])
            if row["home_win"]==1: gs["wins"]+=1
        if away_gid:
            gs=self.goalie_stats[away_gid]; gs["games"]+=1
            gs["shots_against"]+=row["home_sog"]; gs["goals_against"]+=row["home_goals"]
            gs["saves"]+=max(0,row["home_sog"]-row["home_goals"])
            if row["home_win"]==0: gs["wins"]+=1

        # Track last game date
        if "date" in row:
            self.last_game_date[home] = row["date"]
            self.last_game_date[away] = row["date"]

        return predicted, correct

    def train_on_season(self, filepath, base_lr=0.03, reset=True):
        """Train on a full season CSV file."""
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["home_starter_id","away_starter_id"]:
            if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
        for col in ["home_fo_pct","away_fo_pct"]:
            if col in df.columns:
                if df[col].mean()<1.0: df[col]=df[col]*100
                df[col]=df[col].clip(0,100)
        df = df.sort_values("date").reset_index(drop=True)

        # Rest days
        last_game = {}
        hr=[]; ar=[]
        for _, row in df.iterrows():
            h,a=row["home_team"],row["away_team"]; gd=row["date"]
            hr.append(min((gd-last_game[h]).days,7) if h in last_game else 5)
            ar.append(min((gd-last_game[a]).days,7) if a in last_game else 5)
            last_game[h]=gd; last_game[a]=gd
        df["home_rest"]=hr; df["away_rest"]=ar

        if reset:
            all_teams = sorted(set(df["home_team"].unique())|set(df["away_team"].unique()))
            self.reset_season_stats(all_teams)

        correct = 0; total = 0
        for _, row in df.iterrows():
            predicted, was_correct = self.train_on_game(row, base_lr)
            if predicted:
                total += 1
                if was_correct: correct += 1

        acc = correct/total*100 if total > 0 else 0
        return {"accuracy": acc, "predictions": total, "correct": correct}

    def get_rest_days(self, team):
        """Get days of rest for a team based on last known game."""
        if team in self.last_game_date:
            last = self.last_game_date[team]
            if isinstance(last, str):
                last = pd.to_datetime(last)
            days = (datetime.now() - last).days
            return min(days, 7)
        return 3

    def get_likely_starter(self, team, df=None):
        """Get most frequent recent starter for a team."""
        starters = []
        for gid, gs in self.goalie_stats.items():
            # Check if this goalie plays for this team by looking at names
            # This is approximate — in production we'd track team affiliation
            pass

        # Fallback: return the goalie with most games for this team
        # We'd need team-goalie mapping for this to work perfectly
        return None

    def get_weights(self):
        """Return current weights for display."""
        labels = {
            "goal_diff": "Goal Differential", "rest_diff": "Rest Days",
            "give_diff": "Giveaways", "div_flag": "Divisional",
            "b2b_flag": "Back-to-Back", "pk_burden": "PK Burden",
            "goalie_delta": "Goalie Delta", "pp_diff": "Power Play",
            "bias": "Home Ice",
        }
        return [
            {"key": k, "label": labels.get(k, k), "value": round(v, 4)}
            for k, v in sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)
        ]

    def save_state(self, filepath="model_state.json"):
        """Save model state to file."""
        state = {
            "weights": self.weights,
            "goalie_names": {str(k): v for k, v in self.goalie_names.items()},
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath="model_state.json"):
        """Load model state from file."""
        if os.path.exists(filepath):
            with open(filepath) as f:
                state = json.load(f)
            self.weights = state.get("weights", self.weights)
            self.goalie_names = {int(k): v for k, v in state.get("goalie_names", {}).items()}
            return True
        return False

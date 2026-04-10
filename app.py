import csv
import io
from flask import Flask, jsonify, render_template, request, Response
from analysis import load_all

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/magazines")
def api_magazines():
    cache = load_all()
    mags = cache["magazines"]
    result = []
    for name, data in mags.items():
        result.append({
            "name": name,
            "category": data["category"],
            "forecast_2026": data["forecast_2026"],
            "forecast_2027": data["forecast_2027"],
            "sevk_forecast_2026": data.get("sevk_forecast_2026"),
            "iade_forecast_2026": data.get("iade_forecast_2026"),
            "avg_basari": data["avg_basari"],
            "change_rate_last": data["change_rate_last"],
            "last_net": data["net_actual"][-1] if data["net_actual"] else None,
            "last_year": data["years_actual"][-1] if data["years_actual"] else None,
            "best_model": data.get("best_model"),
            "model_score": data.get("model_score"),
            "risk_level": data.get("risk_level"),
            "risk_score": data.get("risk_score"),
        })
    result.sort(key=lambda x: x["forecast_2026"] or 0, reverse=True)
    return jsonify(result)


@app.route("/api/analyze")
def api_analyze():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "Dergi adı gerekli"}), 400

    cache = load_all()
    mags = cache["magazines"]

    if name not in mags:
        matched = next((k for k in mags if k.lower() == name.lower()), None)
        if not matched:
            return jsonify({"error": f"'{name}' bulunamadı"}), 404
        name = matched

    data = mags[name]
    national = cache["national_tiraj"]

    return jsonify({
        "name": data["name"],
        "category": data["category"],
        "years_actual": data["years_actual"],
        "sevk_actual": data["sevk_actual"],
        "iade_actual": data["iade_actual"],
        "net_actual": data["net_actual"],
        # Net projeksiyonu (mevcut)
        "forecast_2026": data["forecast_2026"],
        "forecast_2027": data["forecast_2027"],
        # Sevk/iade projeksiyonları
        "sevk_forecast_2026": data.get("sevk_forecast_2026"),
        "sevk_forecast_2027": data.get("sevk_forecast_2027"),
        "iade_forecast_2026": data.get("iade_forecast_2026"),
        "iade_forecast_2027": data.get("iade_forecast_2027"),
        "net_forecast_2026": data.get("net_forecast_2026"),
        "net_forecast_2027": data.get("net_forecast_2027"),
        # Güven aralıkları
        "ci_low_2026": data.get("ci_low_2026"),
        "ci_high_2026": data.get("ci_high_2026"),
        "ci_low_2027": data.get("ci_low_2027"),
        "ci_high_2027": data.get("ci_high_2027"),
        # Başarı oranı
        "basari_forecast_2026": data.get("basari_forecast_2026"),
        "basari_forecast_2027": data.get("basari_forecast_2027"),
        # Model bilgisi
        "best_model": data.get("best_model"),
        "model_score": data.get("model_score"),
        # Diğer
        "avg_basari": data["avg_basari"],
        "change_rate_last": data["change_rate_last"],
        "risk_level": data.get("risk_level"),
        "risk_score": data.get("risk_score"),
        "iade_ratio_trend": data.get("iade_ratio_trend"),
        "national_tiraj": {str(k): v for k, v in national.items()},
    })


@app.route("/api/summary")
def api_summary():
    cache = load_all()
    mags = cache["magazines"]
    national = cache["national_tiraj"]

    # Yıllık toplam satış (2019-2027)
    years_range = list(range(2019, 2026))
    total_by_year = {}
    for yr in years_range:
        total_by_year[yr] = round(sum(
            (d["net_actual"][d["years_actual"].index(yr)] if yr in d["years_actual"] else 0)
            for d in mags.values()
        ), 0)
    total_by_year[2026] = round(sum(d["forecast_2026"] or 0 for d in mags.values()), 0)
    total_by_year[2027] = round(sum(d["forecast_2027"] or 0 for d in mags.values()), 0)

    total_net_2024 = sum(
        (d["net_actual"][d["years_actual"].index(2024)]
         if 2024 in d["years_actual"] else 0)
        for d in mags.values()
    )
    total_net_2025 = sum(
        (d["net_actual"][d["years_actual"].index(2025)]
         if 2025 in d["years_actual"] else 0)
        for d in mags.values()
    )
    total_forecast_2026 = sum(d["forecast_2026"] or 0 for d in mags.values())
    total_forecast_2027 = sum(d["forecast_2027"] or 0 for d in mags.values())

    by_category = {}
    for d in mags.values():
        cat = d["category"]
        by_category.setdefault(cat, {"count": 0, "forecast_2026": 0, "forecast_2027": 0})
        by_category[cat]["count"] += 1
        by_category[cat]["forecast_2026"] += d["forecast_2026"] or 0
        by_category[cat]["forecast_2027"] += d["forecast_2027"] or 0

    top10_2026 = sorted(
        [{"name": n, "forecast_2026": d["forecast_2026"], "category": d["category"],
          "risk_level": d.get("risk_level")}
         for n, d in mags.items() if d["forecast_2026"]],
        key=lambda x: x["forecast_2026"],
        reverse=True
    )[:10]

    return jsonify({
        "total_magazines": len(mags),
        "total_net_2024": round(total_net_2024, 0),
        "total_net_2025": round(total_net_2025, 0),
        "total_forecast_2026": round(total_forecast_2026, 0),
        "total_forecast_2027": round(total_forecast_2027, 0),
        "by_category": by_category,
        "top10_2026": top10_2026,
        "total_by_year": {str(k): v for k, v in total_by_year.items()},
        "national_tiraj": {str(k): v for k, v in national.items()},
    })


@app.route("/api/compare")
def api_compare():
    """2-4 derginin net trend ve projeksiyonlarını yan yana döner."""
    names_raw = request.args.get("names", "")
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if not names:
        return jsonify({"error": "En az bir dergi adı gerekli"}), 400

    cache = load_all()
    mags = cache["magazines"]

    result = []
    for name in names[:4]:
        matched = next((k for k in mags if k.lower() == name.lower()), None)
        if not matched:
            continue
        d = mags[matched]
        all_years = d["years_actual"] + [2026, 2027]
        net_all = d["net_actual"] + [d["forecast_2026"], d["forecast_2027"]]
        result.append({
            "name": matched,
            "category": d["category"],
            "years": all_years,
            "net": net_all,
            "forecast_start_idx": len(d["years_actual"]),
            "ci_low_2026": d.get("ci_low_2026"),
            "ci_high_2026": d.get("ci_high_2026"),
            "best_model": d.get("best_model"),
            "risk_level": d.get("risk_level"),
        })

    return jsonify(result)


@app.route("/api/export")
def api_export():
    """Bir dergi için CSV indirme."""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "Dergi adı gerekli"}), 400

    cache = load_all()
    mags = cache["magazines"]
    matched = next((k for k in mags if k.lower() == name.lower()), None)
    if not matched:
        return jsonify({"error": "Bulunamadı"}), 404

    d = mags[matched]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Yıl", "Sevk", "İade", "Net Satış", "Tür", "Sevk Tahmin", "İade Tahmin", "Net Tahmin", "CI Alt", "CI Üst"])

    for i, y in enumerate(d["years_actual"]):
        writer.writerow([y, d["sevk_actual"][i], d["iade_actual"][i], d["net_actual"][i], "Gerçek", "", "", "", "", ""])

    writer.writerow([2026, "", "", "", "Projeksiyon",
                     d.get("sevk_forecast_2026", ""), d.get("iade_forecast_2026", ""),
                     d.get("net_forecast_2026", ""), d.get("ci_low_2026", ""), d.get("ci_high_2026", "")])
    writer.writerow([2027, "", "", "", "Projeksiyon",
                     d.get("sevk_forecast_2027", ""), d.get("iade_forecast_2027", ""),
                     d.get("net_forecast_2027", ""), d.get("ci_low_2027", ""), d.get("ci_high_2027", "")])

    output.seek(0)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in matched)
    return Response(
        "\ufeff" + output.getvalue(),  # BOM for Excel UTF-8
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_projeksiyon.csv"'}
    )


@app.route("/api/risk")
def api_risk():
    """Tüm dergileri risk skoruna göre sıralar."""
    cache = load_all()
    mags = cache["magazines"]

    risk_list = sorted(
        [{"name": n, "category": d["category"], "risk_score": d.get("risk_score", 0),
          "risk_level": d.get("risk_level", "Orta"),
          "forecast_2026": d["forecast_2026"],
          "best_model": d.get("best_model"),
          "iade_ratio_trend": d.get("iade_ratio_trend")}
         for n, d in mags.items()],
        key=lambda x: x["risk_score"] or 0,
        reverse=True
    )

    counts = {"Yüksek": 0, "Orta": 0, "Düşük": 0}
    for r in risk_list:
        lvl = r["risk_level"]
        if lvl in counts:
            counts[lvl] += 1

    return jsonify({"risk_list": risk_list, "counts": counts})


@app.route("/api/reload")
def api_reload():
    cache = load_all(force=True)
    return jsonify({"status": "ok", "magazine_count": len(cache["magazines"])})


@app.route("/api/cv_metrics")
def api_cv_metrics():
    """Walk-forward CV metrikleri + feature importance."""
    import numpy as np
    cache = load_all()

    def _to_native(obj):
        """numpy int/float → Python native dönüştür."""
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    cv = _to_native(cache.get("cv_metrics", {}))
    return jsonify({
        # Net Sevk CV metrikleri
        "cv_metrics":         cv,
        # Toplam Sevk CV metrikleri (ayrı model)
        "sevk_cv": {
            "folds":     cv.get("sevk_folds", []),
            "mean_mae":  cv.get("sevk_mean_mae"),
            "mean_mape": cv.get("sevk_mean_mape"),
            "mean_r2":   cv.get("sevk_mean_r2"),
        },
        "feature_importance": _to_native(cache.get("feature_importance", [])),
        # Not: iade_forecast = sevk_forecast - net_forecast (türetilmiş, model yok)
        "iade_method": "derived",
    })


if __name__ == "__main__":
    print("Sunucu başlatılıyor: http://127.0.0.1:5000")
    print("(Veriler ilk API isteğinde yüklenecek ~10 sn)")
    app.run(debug=False, port=5001)

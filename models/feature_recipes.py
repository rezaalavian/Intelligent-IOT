_BASE = ["temp definition °c", "dew point definition °c", "rel hum definition %",
         "wind_u", "wind_v", "pm25"]
RECIPES = {
    "base6": list(_BASE),
    "diffusion9": _BASE + ["upwind_pm25", "transport_potential", "wind_alignment"],
    "base+upwind": _BASE + ["upwind_pm25"],
    "base+transport": _BASE + ["transport_potential"],
    "base+alignment": _BASE + ["wind_alignment"],
}
_GASES = ["no", "no2", "nox", "o3"]
RECIPES["base+pollutants"] = RECIPES["base6"] + _GASES
RECIPES["with_pollutants"] = RECIPES["diffusion9"] + _GASES

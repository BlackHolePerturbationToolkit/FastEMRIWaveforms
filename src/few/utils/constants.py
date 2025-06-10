from lisaconstants import (
    ASTRONOMICAL_YEAR,
    GM_SUN,
    PARSEC,
    SPEED_OF_LIGHT,
    SUN_SCHWARZSCHILD_RADIUS,
)

YRSID_SI = ASTRONOMICAL_YEAR  # Number of seconds in one astronomical year
MTSUN_SI = GM_SUN / (SPEED_OF_LIGHT**3)  # One solar mass in seconds
MRSUN_SI = 0.5 * SUN_SCHWARZSCHILD_RADIUS  # Schwarzschild radius in meters
Gpc = 1e9 * PARSEC  # One Gpc in meters
PI = 3.141592653589793238462643383279502884  # Value of Pi

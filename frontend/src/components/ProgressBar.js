const ProgressBar = ({
    progressBarRef,
    audioRefs,
    playing,
    trackNumber
}) => {
    const handleProgressChange = () => {
        audioRefs.current[playing].currentTime = progressBarRef.current.value;
    };

    const formatTime = (time) => {
        if (time && !isNaN(time)) {
            const minutes = Math.floor(time / 60);
            const formatMinutes =
                minutes < 10 ? `0${minutes}` : `${minutes}`;
            const seconds = Math.floor(time % 60);
            const formatSeconds =
                seconds < 10 ? `0${seconds}` : `${seconds}`;
            return `${formatMinutes}:${formatSeconds}`;
        }
        return '00:00';
    };
    let timeProgress = 0;
    let duration = 3;
    return (
        <div className="progress">
            <span className="time current">{formatTime(timeProgress)}</span>
            <input
                type="range"
                ref={progressBarRef}
                defaultValue="0"
                onChange={handleProgressChange}
            />
            <span className="time">{formatTime(duration)}</span>
        </div>
    );
};

export default ProgressBar;
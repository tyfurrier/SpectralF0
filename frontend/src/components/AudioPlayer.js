import { useState, useRef } from 'react';
import getCombo from '../data/sounds';

// import components
import DisplayTrack from './DisplayAudio';
import Controls from './Controls';
import ProgressBar from './ProgressBar';

const AudioPlayer = ({ playing, setPlaying }) => {
    const [tracks, setTracks] = useState(getCombo());
    console.log(tracks);
    console.log(tracks.length);
    const audioRefs = useRef(new Array(tracks.length));
    const progressBarRefs = useRef([]);
    let divs = [];
    for (let i = 0; i < tracks.length; i++) {
        divs.push(
            <div className="audio-player">
                <div className="inner">
                    <p>{playing}</p>
                    {/* <p>{audioRefs.current[i]}</p> */}
                    <p>{tracks[i]}</p>
                    <DisplayTrack
                        {...{
                            tracks,
                            audioRefs,
                            playing,
                            progressBarRefs,
                            setPlaying,
                            trackNumber: i
                        }}
                    />
                    <Controls
                        {...{
                            playing,
                            audioRefs,
                            setPlaying,
                            trackNumber: i
                        }}
                    />
                    {/* <ProgressBar
                        {...{
                            progressBarRefs,
                            audioRefs,
                            playing,
                            trackNumber: i
                        }}
                    /> */}
                </div>
            </div>
        )
    }
    return (
        <div className="audio-player">
            {divs}
        </div>
    );
};
export default AudioPlayer;
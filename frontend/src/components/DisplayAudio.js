import { BsMusicNoteBeamed } from 'react-icons/bs';
import { BellIcon, PhoneIcon } from "@chakra-ui/icons";
import { IconButton } from '@chakra-ui/react'
import { useEffect, useState } from 'react';

function fNameToPath(fname) {
    return `/${fname}`
}

const DisplayTrack = ({
    tracks,
    audioRefs,
    playing,
    setPlaying,
    trackNumber }) => {
    const [active, setActive] = useState(false);
    useEffect(() => {
        if (playing == trackNumber) {
            console.log(`audio refs: ${audioRefs}`)
            setActive(true);
        } else {
            setActive(false);
        }
    }, [playing]);



    console.log(playing);
    const handleClick = () => {
        setPlaying(trackNumber);
    }
    console.log(playing);

    return (
        <div>
            <audio
                src={fNameToPath(tracks[trackNumber])}
                ref={(element) => { audioRefs.current[trackNumber] = element }}
                key={trackNumber}
            />
            <div className="audio-info">
                <div className="audio-image">
                    <div className="icon-wrapper">
                        <span className="audio-icon">

                        </span>
                    </div>
                </div>
                <div className="text">
                    <button
                        onClick={handleClick}>
                        {`Sound ${trackNumber}`}
                    </button>
                </div>
            </div>
        </div>
    );
};
export default DisplayTrack;
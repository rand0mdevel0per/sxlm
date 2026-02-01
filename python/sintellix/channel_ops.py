"""
Channel Operations API for Sintellix

This module provides Python API for extracting subchannels from model output,
wrapping them with text/img/audio shells, and inserting into NMDB auxiliary channels.

Critical for saltts project integration.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import struct


class ChannelType:
    """Channel type constants matching nmdb proto definitions"""
    UNSPECIFIED = 0
    TEXT = 1
    TTS = 2
    IMAGE = 3
    AUDIO = 4
    CUSTOM = 5
    CONTROL = 6
    TEXT_AUX = 7
    IMAGE_AUX = 8
    AUDIO_AUX = 9


class SubchannelExtractor:
    """
    Extract subchannels from sintellix model output.

    This class provides methods to extract specific subchannels from
    a model's output channel for further processing or insertion into
    auxiliary channels.
    """

    def __init__(self):
        pass

    def extract_subchannel(
        self,
        output_data: np.ndarray,
        start_idx: int,
        length: int,
        channel_name: str = "subchannel"
    ) -> Dict:
        """
        Extract a subchannel from model output data.

        Args:
            output_data: Model output as numpy array
            start_idx: Starting index for extraction
            length: Length of subchannel to extract
            channel_name: Name for the extracted subchannel

        Returns:
            Dictionary containing subchannel data and metadata
        """
        if start_idx < 0 or start_idx + length > len(output_data):
            raise ValueError(f"Invalid subchannel range: [{start_idx}, {start_idx + length})")

        subchannel_data = output_data[start_idx:start_idx + length]

        return {
            'name': channel_name,
            'data': subchannel_data,
            'dimension': length,
            'start_idx': start_idx,
            'metadata': {
                'source': 'sintellix_model_output',
                'extraction_range': f'[{start_idx}:{start_idx + length}]'
            }
        }

    def extract_multiple_subchannels(
        self,
        output_data: np.ndarray,
        ranges: List[Tuple[int, int, str]]
    ) -> List[Dict]:
        """
        Extract multiple subchannels from model output.

        Args:
            output_data: Model output as numpy array
            ranges: List of (start_idx, length, name) tuples

        Returns:
            List of subchannel dictionaries
        """
        subchannels = []
        for start_idx, length, name in ranges:
            subchannel = self.extract_subchannel(output_data, start_idx, length, name)
            subchannels.append(subchannel)

        return subchannels


class ChannelWrapper:
    """
    Wrap subchannels with text/image/audio shells for auxiliary channel insertion.

    This class provides methods to wrap extracted subchannels with appropriate
    type-specific shells (text, image, or audio) for insertion into NMDB
    auxiliary channels.
    """

    def __init__(self):
        pass

    def wrap_as_text(
        self,
        subchannel: Dict,
        encoding: str = "utf-8",
        language: str = "en"
    ) -> Dict:
        """
        Wrap subchannel with text shell for TEXT_AUX channel.

        Args:
            subchannel: Subchannel dictionary from SubchannelExtractor
            encoding: Text encoding (default: utf-8)
            language: Language code (default: en)

        Returns:
            Wrapped channel ready for TEXT_AUX insertion
        """
        data = subchannel['data']

        # Serialize numpy array to bytes
        data_bytes = data.tobytes()

        return {
            'name': subchannel['name'] + '_text',
            'type': ChannelType.TEXT_AUX,
            'data': data_bytes,
            'dimension': subchannel['dimension'],
            'metadata': {
                **subchannel.get('metadata', {}),
                'encoding': encoding,
                'language': language,
                'wrapper_type': 'text',
                'original_dtype': str(data.dtype),
                'original_shape': str(data.shape)
            }
        }

    def wrap_as_image(
        self,
        subchannel: Dict,
        format: str = "raw",
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict:
        """
        Wrap subchannel with image shell for IMAGE_AUX channel.

        Args:
            subchannel: Subchannel dictionary from SubchannelExtractor
            format: Image format (raw, png, jpeg, etc.)
            width: Image width (if applicable)
            height: Image height (if applicable)

        Returns:
            Wrapped channel ready for IMAGE_AUX insertion
        """
        data = subchannel['data']
        data_bytes = data.tobytes()

        metadata = {
            **subchannel.get('metadata', {}),
            'format': format,
            'wrapper_type': 'image',
            'original_dtype': str(data.dtype),
            'original_shape': str(data.shape)
        }

        if width is not None:
            metadata['width'] = str(width)
        if height is not None:
            metadata['height'] = str(height)

        return {
            'name': subchannel['name'] + '_image',
            'type': ChannelType.IMAGE_AUX,
            'data': data_bytes,
            'dimension': subchannel['dimension'],
            'metadata': metadata
        }

    def wrap_as_audio(
        self,
        subchannel: Dict,
        format: str = "raw",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16
    ) -> Dict:
        """
        Wrap subchannel with audio shell for AUDIO_AUX channel.

        Args:
            subchannel: Subchannel dictionary from SubchannelExtractor
            format: Audio format (raw, wav, mp3, etc.)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            bit_depth: Bit depth

        Returns:
            Wrapped channel ready for AUDIO_AUX insertion
        """
        data = subchannel['data']
        data_bytes = data.tobytes()

        return {
            'name': subchannel['name'] + '_audio',
            'type': ChannelType.AUDIO_AUX,
            'data': data_bytes,
            'dimension': subchannel['dimension'],
            'metadata': {
                **subchannel.get('metadata', {}),
                'format': format,
                'sample_rate': str(sample_rate),
                'channels': str(channels),
                'bit_depth': str(bit_depth),
                'wrapper_type': 'audio',
                'original_dtype': str(data.dtype),
                'original_shape': str(data.shape)
            }
        }


class AuxiliaryChannelInserter:
    """
    Insert wrapped channels into NMDB auxiliary channels.

    This class provides methods to insert wrapped channels into NMDB
    auxiliary channels for communication with external components.
    """

    def __init__(self, nmdb_connection=None):
        """
        Initialize the inserter with optional NMDB connection.

        Args:
            nmdb_connection: Optional NMDB connection object
        """
        self.nmdb_connection = nmdb_connection

    def insert_to_aux_channel(
        self,
        wrapped_channel: Dict,
        channel_name: str
    ) -> bool:
        """
        Insert wrapped channel into NMDB auxiliary channel.

        Args:
            wrapped_channel: Wrapped channel from ChannelWrapper
            channel_name: Target auxiliary channel name

        Returns:
            True if insertion successful, False otherwise
        """
        if self.nmdb_connection is None:
            # Placeholder for actual NMDB connection
            # In production, this would use the NMDB client library
            print(f"[Placeholder] Inserting channel '{wrapped_channel['name']}' "
                  f"into auxiliary channel '{channel_name}'")
            return True

        # Actual NMDB insertion would go here
        # self.nmdb_connection.store_peripheral(channel_name, wrapped_channel['name'], wrapped_channel)
        return True


# High-level convenience functions

def extract_and_wrap_text(
    model_output: np.ndarray,
    start_idx: int,
    length: int,
    name: str = "text_subchannel",
    encoding: str = "utf-8",
    language: str = "en"
) -> Dict:
    """
    Convenience function: Extract subchannel and wrap as text in one step.

    Args:
        model_output: Model output array
        start_idx: Starting index for extraction
        length: Length of subchannel
        name: Channel name
        encoding: Text encoding
        language: Language code

    Returns:
        Wrapped text channel ready for insertion
    """
    extractor = SubchannelExtractor()
    wrapper = ChannelWrapper()

    subchannel = extractor.extract_subchannel(model_output, start_idx, length, name)
    return wrapper.wrap_as_text(subchannel, encoding, language)


def extract_and_wrap_audio(
    model_output: np.ndarray,
    start_idx: int,
    length: int,
    name: str = "audio_subchannel",
    sample_rate: int = 16000,
    channels: int = 1
) -> Dict:
    """
    Convenience function: Extract subchannel and wrap as audio in one step.

    Args:
        model_output: Model output array
        start_idx: Starting index for extraction
        length: Length of subchannel
        name: Channel name
        sample_rate: Audio sample rate
        channels: Number of audio channels

    Returns:
        Wrapped audio channel ready for insertion
    """
    extractor = SubchannelExtractor()
    wrapper = ChannelWrapper()

    subchannel = extractor.extract_subchannel(model_output, start_idx, length, name)
    return wrapper.wrap_as_audio(subchannel, sample_rate=sample_rate, channels=channels)


def extract_wrap_and_insert(
    model_output: np.ndarray,
    start_idx: int,
    length: int,
    wrapper_type: str,
    aux_channel_name: str,
    nmdb_connection=None,
    **kwargs
) -> bool:
    """
    Complete pipeline: Extract, wrap, and insert into auxiliary channel.

    This is the main convenience function for the saltts project workflow.

    Args:
        model_output: Model output array
        start_idx: Starting index for extraction
        length: Length of subchannel
        wrapper_type: Type of wrapper ('text', 'image', or 'audio')
        aux_channel_name: Target auxiliary channel name
        nmdb_connection: Optional NMDB connection
        **kwargs: Additional arguments for wrapper

    Returns:
        True if successful, False otherwise

    Example:
        >>> # Extract audio subchannel and insert into audio auxiliary channel
        >>> success = extract_wrap_and_insert(
        ...     model_output=output_array,
        ...     start_idx=0,
        ...     length=1024,
        ...     wrapper_type='audio',
        ...     aux_channel_name='audio_aux',
        ...     sample_rate=22050
        ... )
    """
    extractor = SubchannelExtractor()
    wrapper = ChannelWrapper()
    inserter = AuxiliaryChannelInserter(nmdb_connection)

    # Extract subchannel
    subchannel = extractor.extract_subchannel(
        model_output, start_idx, length,
        kwargs.get('name', f'{wrapper_type}_subchannel')
    )

    # Wrap based on type
    if wrapper_type == 'text':
        wrapped = wrapper.wrap_as_text(subchannel, **kwargs)
    elif wrapper_type == 'image':
        wrapped = wrapper.wrap_as_image(subchannel, **kwargs)
    elif wrapper_type == 'audio':
        wrapped = wrapper.wrap_as_audio(subchannel, **kwargs)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    # Insert into auxiliary channel
    return inserter.insert_to_aux_channel(wrapped, aux_channel_name)


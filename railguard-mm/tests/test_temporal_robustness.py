import torch

from ml.evaluate_multimodal import hold_last_frame_dropout, shift_frames


def _frames():
    # [B,T,C,H,W], values make temporal identity easy to inspect.
    x=torch.arange(5,dtype=torch.float32).reshape(1,5,1,1,1)
    return x


def test_frame_shift_never_wraps_future_or_past_samples():
    x=_frames()
    assert shift_frames(x,2).flatten().tolist()==[0,0,0,1,2]
    assert shift_frames(x,-2).flatten().tolist()==[2,3,4,4,4]


def test_hold_last_dropout_is_reproducible_and_preserves_first_anchor():
    x=_frames()
    a=hold_last_frame_dropout(x,0.5,torch.Generator().manual_seed(4))
    b=hold_last_frame_dropout(x,0.5,torch.Generator().manual_seed(4))
    assert torch.equal(a,b)
    assert a[:,0].equal(x[:,0])
    # Dropout can only hold the previous output; it cannot introduce a future value.
    for t in range(1,x.shape[1]):
        assert float(a[0,t]) <= float(x[0,t])

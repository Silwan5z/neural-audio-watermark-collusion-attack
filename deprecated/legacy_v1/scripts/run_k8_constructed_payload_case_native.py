#!/usr/bin/env python3
"""Native-sample-rate K=8 constructed-payload validation for five systems."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from registry import NBITS, full_registry_bits, load_clean  # noqa: E402
from watermarks import (  # noqa: E402
    _chunk_logits_to_bit_evidence, detect_many, embed, extract_evidence,
    get_timbrewm, get_wavmark, get_wmcodec, pesq_wb, resample_to, stoi,
)
from run_k8_constructed_payload_case import BITS16, COUNTS16, SEED, validate_design  # noqa: E402

MODELS=("audioseal","wavmark","timbrewm","voicemark","wmcodec")
NATIVE_SR={"audioseal":16000,"wavmark":16000,"voicemark":16000,
           "timbrewm":22050,"wmcodec":24000}
OUT=ROOT/"data"/"k8_constructed_payload_case_native_20260829"


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=MODELS)
    p.add_argument("--speaker",default="chinese:SSB0005")
    p.add_argument("--output-dir",type=Path,default=OUT)
    return p.parse_args()


def bits_to_int(bits): return int(sum(int(v)<<i for i,v in enumerate(bits)))


def embed_native(model,clean16,bits):
    if model not in ("timbrewm","wmcodec"):
        return embed(model,clean16,bits),16000
    import torch
    if model=="timbrewm":
        m=get_timbrewm(); clean=resample_to(clean16,16000,22050)
        t=torch.from_numpy(clean).float().to(m["dev"])[None,None]
        msg=torch.tensor([[bits]],dtype=torch.float32,device=m["dev"])*2-1
        with torch.no_grad(): y,_=m["enc"].test_forward(t,msg)
        return y[0,0].cpu().numpy().astype(np.float32),22050
    m=get_wmcodec(); clean=resample_to(clean16,16000,24000)
    t=torch.from_numpy(clean).float().to(m["dev"])[None,None]
    digits=[int("".join(map(str,bits[k*4:(k+1)*4])),2) for k in range(4)]
    sign=torch.tensor([digits],dtype=torch.long,device=m["dev"])
    with torch.no_grad():
        sign_en=m["wm_enc"](sign); en=m["enc"](t,sign_en); q,_,_=m["quant"](en); y=m["gen"](q)
    return y[0,0].cpu().numpy().astype(np.float32),24000


def decode_native(model,wav,sr):
    """Return hard bits, bit marginals, presence, and native digit diagnostics."""
    import torch
    extra={}
    if model=="timbrewm":
        m=get_timbrewm(); t=torch.from_numpy(wav).float().to(m["dev"])[None,None]
        with torch.no_grad(): soft=m["dec"].test_forward(t)[0,0]
        soft=soft.cpu().numpy(); p1=1/(1+np.exp(-soft)); hard=(soft>=0).astype(np.int8)
        return hard,p1,float("nan"),extra
    if model=="wmcodec":
        from third_party.wmcodec.meldataset import mel_spectrogram
        m=get_wmcodec(); h=m["h"]; t=torch.from_numpy(wav).float().to(m["dev"])[None]
        with torch.no_grad():
            mel=mel_spectrogram(t,h.n_fft,h.num_mels,h.sampling_rate,h.hop_size,
                                h.win_size,h.fmin,h.fmax_for_loss)
            scores,pred=m["wm_dec"](mel)
        digit_probs=torch.softmax(torch.stack(scores,dim=1),dim=-1)[0].cpu().numpy()
        digits=pred[0].cpu().numpy().astype(int).tolist()
        hard=[]
        for v in digits: hard.extend([(v>>(3-j))&1 for j in range(4)])
        evidence=_chunk_logits_to_bit_evidence(digit_probs,bit_order="msb")
        extra={"decoded_digits":digits,"digit_probabilities":digit_probs.tolist(),
               "digit_confidences":[float(digit_probs[k,digits[k]]) for k in range(4)]}
        return np.asarray(hard,dtype=np.int8),(evidence+1)/2,float("nan"),extra
    # Native rate is 16 kHz for the remaining systems; reuse the audited wrappers.
    registry=full_registry_bits(model); _,presence,hard=detect_many(model,[wav],registry)[0]
    if hard is None: return None,None,presence,extra
    if model=="wavmark":
        p1,valid,total=wavmark_vote_probability(wav)
        extra={"valid_start_pattern_windows":valid,"total_sliding_windows":total}
    else:
        p1=(np.asarray(extract_evidence(model,wav),dtype=float)+1)/2
    return np.asarray(hard,dtype=np.int8),np.asarray(p1,dtype=float),presence,extra


def wavmark_vote_probability(wav):
    import torch
    from wavmark.utils import wm_add_util
    m=get_wavmark(); start=np.asarray(wm_add_util.fix_pattern[:16],dtype=np.int8)
    windows=np.stack([wav[p*800:p*800+16000] for p in range((len(wav)-16000)//800)])
    decoded=[]; batch=int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE","600"))
    for off in range(0,len(windows),batch):
        with torch.no_grad(): x=(m["model"].decode(torch.from_numpy(windows[off:off+batch]).to(m["dev"]))>=.5)
        decoded.append(x.int().cpu().numpy())
    decoded=np.concatenate(decoded); keep=np.all(decoded[:,:16]==start[None,:],axis=1)
    if not keep.any(): raise RuntimeError("WavMark: no valid start-pattern window")
    return decoded[keep,16:32].mean(0),int(keep.sum()),int(len(decoded))


def main():
    a=parse_args(); validate_design(); model=a.model; d=NBITS[model]; bits=BITS16[:,:d].copy()
    payloads=[bits_to_int(r) for r in bits]; sr=NATIVE_SR[model]
    audio_dir=a.output_dir/"audio"/model; raw_dir=a.output_dir/"raw"
    audio_dir.mkdir(parents=True,exist_ok=True); raw_dir.mkdir(parents=True,exist_ok=True)
    start=time.time(); clean16=np.asarray(load_clean(a.speaker),dtype=np.float32)
    clean_native=resample_to(clean16,16000,sr) if sr!=16000 else clean16
    sf.write(audio_dir/f"clean_{sr}hz.wav",clean_native,sr,subtype="FLOAT")
    members=[]; paths=[]
    for i,row in enumerate(bits,1):
        wav,got_sr=embed_native(model,clean16,row.tolist())
        if got_sr!=sr or not np.isfinite(wav).all(): raise ValueError(f"bad member {i}")
        members.append(wav)
    n=min(len(x) for x in members); members=[x[:n] for x in members]
    for i,wav in enumerate(members,1):
        p=audio_dir/f"colluder_{i:02d}_payload_{payloads[i-1]}_{sr}hz.wav"
        sf.write(p,wav,sr,subtype="FLOAT"); paths.append(str(p))
    mixed=np.mean(np.stack(members),axis=0,dtype=np.float64).astype(np.float32)
    mix_path=audio_dir/f"equal_mean_k8_{sr}hz.wav"; sf.write(mix_path,mixed,sr,subtype="FLOAT")

    clean_results=[]
    for i,wav in enumerate(members):
        hard,p1,pres,extra=decode_native(model,wav,sr)
        decoded=None if hard is None else bits_to_int(hard)
        clean_results.append({"colluder_index":i+1,"target_payload":payloads[i],
                              "decoded_payload":decoded,"exact":int(decoded==payloads[i]),
                              "presence":None if not np.isfinite(pres) else float(pres),**extra})
    hard,p1,pres,extra=decode_native(model,mixed,sr)
    if hard is None: raise RuntimeError(f"{model}: mixed decode absent")
    counts=bits.sum(0).astype(int); majority=np.where(counts>4,1,np.where(counts<4,0,-1))
    strict=np.where(majority>=0)[0]; ties=np.where(majority<0)[0]
    departures=[int(j) for j in strict if hard[j]!=majority[j]]
    unanimous=[int(j) for j in range(d) if counts[j] in (0,8)]
    uv=[j for j in unanimous if hard[j]!=int(counts[j]==8)]
    agreements=[int(np.count_nonzero(hard==r)) for r in bits]
    decoded=bits_to_int(hard)
    bit_rows=[]
    for j in range(d):
        bit_rows.append({"bit_index_lsb_first":j,"colluder_bits":bits[:,j].astype(int).tolist(),
                         "ones_among_8":int(counts[j]),"state":"tie_4_4" if counts[j]==4 else
                         "unanimous_0" if counts[j]==0 else "unanimous_1" if counts[j]==8 else
                         "strict_majority_1" if counts[j]>4 else "strict_majority_0",
                         "p_bit_1":float(p1[j]),"native_decoded_bit":int(hard[j]),
                         "confidence_native_decoded_bit":float(p1[j] if hard[j] else 1-p1[j])})
    # Quality metrics remain comparable by converting both signals once to 16 kHz.
    ref16=resample_to(members[0],sr,16000) if sr!=16000 else members[0]
    mix16=resample_to(mixed,sr,16000) if sr!=16000 else mixed
    result={"experiment":"native-sample-rate constructed K=8 payload-composition validation",
            "model":model,"K":8,"seed":SEED,"speaker":a.speaker,"native_sample_rate":sr,
            "decode_protocol":"native sample rate; no down-up resampling before attribution",
            "mixing":"equal full-waveform mean","weights":[0.125]*8,"payload_nbits":d,
            "payload_format":"4 base-16 digits plus bijective 16-bit expansion" if model=="wmcodec" else f"{d}-bit",
            "payload_bit_order":"LSB-first canonical identity labels","ones_per_bit":counts.tolist(),
            "payloads":payloads,"payload_bits":bits.astype(int).tolist(),"member_audio_paths":paths,
            "mixed_audio_path":str(mix_path),"clean_audio_path":str(audio_dir/f"clean_{sr}hz.wav"),
            "clean_member_decodes":clean_results,"clean_attribution_exact":sum(x["exact"] for x in clean_results),
            "decoded_identity":decoded,"decoded_bits":hard.astype(int).tolist(),
            "presence":None if not np.isfinite(pres) else float(pres),"bit_results":bit_rows,
            "strict_majority_bits":strict.astype(int).tolist(),"tie_bits":ties.astype(int).tolist(),
            "strict_majority_departures":departures,"unanimous_bits":unanimous,"unanimous_violations":uv,
            "agreement_bits_by_colluder":agreements,"NCA_bits":max(agreements),"NCA":max(agreements)/d,
            "tracing_failure":int(decoded not in payloads),"PESQ_vs_first_copy":pesq_wb(ref16,mix16),
            "STOI_vs_first_copy":stoi(ref16,mix16),"quality_metric_rate":16000,
            "elapsed_seconds":time.time()-start,**extra}
    tmp=raw_dir/f"{model}.json.tmp"; final=raw_dir/f"{model}.json"
    tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); os.replace(tmp,final)
    print(json.dumps({"model":model,"sr":sr,"output":str(final),"clean_exact":result["clean_attribution_exact"],
                      "decoded":decoded,"NCA":result["NCA"],"TF":result["tracing_failure"],
                      "elapsed":result["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__": main()

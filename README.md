# SPHProject
**CUDA SPH Fluid Simulation (Unreal Engine 5.7)**

NVIDIA CUDA로 GPU 병렬 처리하는 **SPH(Smoothed Particle Hydrodynamics) 유체 시뮬레이션**을
언리얼 엔진에서 시각화하는 프로젝트

`bf-sph-ue4/`: UE4.27 + CUDA 11.5로 만든 2021년 버전  
루트: **UE5.7 + CUDA 13.2** 2026년 업데이트 버전

## 구조

```
SPHProject.uproject        UE5.7 프로젝트
Source/SPHProject/         UE 모듈 (ASPHFluidActor = 시뮬 + 렌더)
CudaLibrary/               CUDA 정적 라이브러리 (SPH 커널, 별도 빌드 release)
bf-sph-ue4/                UE4.27 프로젝트 아카이브
```


## 빌드 & 실행

1. **CUDA 라이브러리** — `CudaLibrary/CudaLibrary.vcxproj`를 VS 2022에서 `Release|x64`로 빌드
   → `x64/Release/CudaLibrary.lib` 생성
2. **언리얼 모듈** — 에디터를 닫은 상태에서 `SPHProjectEditor` 타깃을 빌드
3. 에디터에서 `ASPHFluidActor`를 레벨에 배치하여 테스트

### 환경
- GPU: NVIDIA RTX 계열, compute capability **8.9 (sm_89)** 기준
- CUDA Toolkit **13.2**, Visual Studio **2022** (CUDA VS 통합 포함), Unreal Engine **5.7**

## UE4.27 + CUDA 11.5 → UE5.7 + CUDA 13.2 마이그레이션 수정사항

### CUDA 커널 
`CudaLibrary/kernel.cu`
- `entireForces[i] = entireForce` (float3 → float4 슬롯 대입 불가) → `make_float4(entireForce, 0.0f)`
- `integrate_functor`에서 `volatile float4` 제거 — 최신 thrust(CCCL) 참조 타입과 `operator=`
  불일치(*"no operator = matches these operands"*) 해결
- `tables.h` include와 VS 템플릿 잔재(`addKernel`/`addWithCuda`) 제거
- **Marching Cubes 표면 복원 코드 제외** — 정의되지 않은 `gParameters`/`CudaSimParams`를 참조하여
  포팅에서 제외

### 언리얼 액터 
`AMaxim_de_Winter` → `ASPHFluidActor`
- UE5로 포팅. `ProceduralMeshComponent`, marching cubes, 정의 없던 `addSphere/addCollideSphere` 제거.
- 엔진 기본 구체 메시 자동 할당 + 루트 컴포넌트 설정 → 에디터 셋업 없이 바로 렌더.
- `EndPlay`에서 GPU 메모리 해제.
- **성능**: 입자 인스턴스의 그림자/콜리전/내비게이션 비활성화 + 매 프레임
  `MarkRenderStateDirty`(렌더 프록시 재생성) 대신 `BatchUpdateInstancesTransforms` 사용.
- **FPS 독립 타임스텝**: 고정 `0.03/frame`(속도가 프레임레이트에 비례) →
  `dt = min(DeltaTime * SimSpeed, MaxStepTime)`.
- `SimSpeed`, `MaxStepTime`, 시각화 스케일을 `EditAnywhere`로 노출 → 재컴파일 없이 에디터에서 튜닝.

### SPH 안정성 / 버그 수정
- **초기화 폭발 버그**: `resetStart`의 초기 격자 크기가 `s/3`로 너무 작아(예: 500개에 27칸)
  대부분의 유체 입자가 원점(0,0,0)에 겹친 채 시작 → 밀도 특이점 → 폭발. 격자를
  `ceil(유체수^(1/3))`로 수정해 모든 입자를 제대로 배치.
- **경계 입자 고정**: force 커널이 boundary 입자에도 중력을 누적해 가라앉던 문제 →
  boundary 분기에서 속도/힘을 0으로 고정(밀도/압력엔 계속 기여하는 정적 벽).
- **압력 불안정 튜닝**: `pressure = GasStiffness × (density − RestDensity)`에서 RestDensity가
  실제 정지밀도보다 낮으면 전 입자가 양압 → 팽창/폭발. `RestDensity`/`GasStiffness`/`Viscosity`를
  `EditAnywhere`로 노출해 폭발이 잦아들 때까지 에디터에서 실시간 튜닝.

/* Dựng deck tóm tắt nhóm 16 — 23 slide, style theo bản Walnut gốc.
   Chạy: node output/presentation/build_deck.js  (pptxgenjs cài global) */
const path = require('path');
const fs = require('fs');
const pptxgen = require(process.env.GROOT + '/pptxgenjs');

const ROOT = '/Users/trieulh/Documents/Master/20252B_IT5410/Final';
const ASSET = '/tmp/deck_assets';
const fig = (p) => path.join(ROOT, p);
const LOGO = path.join(ROOT, 'output/presentation/assets/hust-logo.png');
const HAS_LOGO = fs.existsSync(LOGO);

// ---- bảng màu / font trích từ deck gốc ----
const C = {
  ink: '000000', gray: '515761', mute: '7A828C',
  blue: '3D8DFF', teal: '156082', orange: 'E97132', green: '196B24', red: 'C0392B',
  line: 'B8BCC4', panel: 'F4F6F9', white: 'FFFFFF',
};
// Arial: có sẵn trên cả macOS (Keynote) và Windows (PowerPoint) -> không bị thay font.
// Arial không có biến thể Light phổ dụng nên tiêu đề lớn dùng chính Arial.
const F = 'Arial', FL = 'Arial';

const p = new pptxgen();
p.defineLayout({ name: 'W', width: 13.333, height: 7.5 });
p.layout = 'W';
let PAGE = 0;

// khung chuẩn cho slide nội dung: eyebrow + title + số trang + logo góc
function chrome(s, title, { eyebrow = 'TINYSTORY-VN · NHÓM 16 · IT5410', titleColor = C.ink, titleSize = 25 } = {}) {
  PAGE += 1;
  s.background = { color: C.white };
  s.addText(eyebrow, { x: 0.44, y: 0.30, w: 8, h: 0.3, fontFace: F, fontSize: 11.5, color: C.gray, charSpacing: 1 });
  s.addText(title, { x: 0.44, y: 0.62, w: 10.7, h: 0.95, fontFace: F, fontSize: titleSize, bold: true, color: titleColor, lineSpacingMultiple: 0.98 });
  s.addText(String(PAGE), { x: 12.5, y: 6.95, w: 0.5, h: 0.3, fontFace: F, fontSize: 10.5, color: C.gray, align: 'right' });
  if (HAS_LOGO) s.addImage({ path: LOGO, x: 11.75, y: 0.26, w: 1.25, h: 0.53, sizing: { type: 'contain', w: 1.25, h: 0.53 } });
}
// danh sách gạch đầu dòng (câu hoàn chỉnh)
function bullets(items) {
  return items.map((t, i) => ({ text: t, options: { bullet: { code: '2022', indent: 14 }, breakLine: true, paraSpaceAfter: 7, fontFace: F, fontSize: 14.5, color: C.ink } }));
}
// khối số liệu nổi bật (số lớn + mô tả)
function stat(s, x, y, big, desc, color = C.blue, w = 4.2) {
  s.addText(big, { x, y, w, h: 0.6, fontFace: FL, fontSize: 30, bold: true, color });
  s.addText(desc, { x, y: y + 0.58, w, h: 0.7, fontFace: F, fontSize: 13.5, color: C.gray, lineSpacingMultiple: 0.98 });
}
function img(s, file, o) { s.addImage({ path: file, ...o, sizing: { type: 'contain', w: o.w, h: o.h } }); }
function vline(s, x, y, h) { s.addShape(p.ShapeType.line, { x, y, w: 0, h, line: { color: C.line, width: 1 } }); }

// ============================ 1. BÌA ============================
{
  const s = p.addSlide(); s.background = { color: C.white };
  PAGE += 1; // bìa là slide 1 (không in số), giữ số trang khớp vị trí
  s.addText('BÁO CÁO TỔNG HỢP · NHÓM 16 · IT5410', { x: 0.7, y: 0.6, w: 9, h: 0.4, fontFace: F, fontSize: 15, color: C.gray, charSpacing: 1 });
  s.addText([
    { text: 'Đánh giá thực nghiệm', options: { breakLine: true } },
    { text: 'các mô hình sinh truyện ngụ ngôn theo điều kiện', options: { breakLine: true } },
  ], { x: 0.7, y: 1.7, w: 11.5, h: 2.2, fontFace: FL, fontSize: 46, bold: true, color: C.ink, lineSpacingMultiple: 1.02 });
  s.addText('Năm hệ thống độc lập · tiền huấn luyện từ đầu · PEFT/QLoRA · kiểm soát đầu ra', { x: 0.7, y: 4.35, w: 11.8, h: 0.5, fontFace: F, fontSize: 20, color: C.gray });
  s.addText([
    { text: '20252611M · Lê Hải Triều      20252612M · Đào Đức Tùng      20252610M · Nguyễn Công Thanh', options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: '20252130M · Nguyễn Thị Phương Liên      20252737M · Nguyễn Đình Lê Hoàng', options: {} },
  ], { x: 0.7, y: 5.5, w: 11.8, h: 1, fontFace: F, fontSize: 13.5, color: C.ink });
  if (HAS_LOGO) s.addImage({ path: LOGO, x: 10.9, y: 0.5, w: 1.9, h: 0.8, sizing: { type: 'contain', w: 1.9, h: 0.8 } });
}

// ============================ 2. ĐỀ TÀI & BÀI TOÁN ============================
{
  const s = p.addSlide(); chrome(s, 'Đề tài và bài toán nghiên cứu');
  s.addText('Tác vụ là sinh truyện ngụ ngôn thiếu nhi bằng tiếng Anh, có điều kiện theo năm trường tường thuật, với yêu cầu chạy được cục bộ. Thách thức trọng tâm không nằm ở độ trôi chảy: một truyện đọc mượt vẫn có thể bỏ qua các điều kiện được yêu cầu. Câu hỏi đặt ra là các điều kiện có thực sự chi phối diễn biến hay chỉ xuất hiện như từ khóa.',
    { x: 0.44, y: 1.75, w: 7.1, h: 1.9, fontFace: F, fontSize: 15, color: C.ink, lineSpacingMultiple: 1.05 });
  s.addText('Bốn câu hỏi nghiên cứu', { x: 0.44, y: 3.75, w: 7, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.teal });
  s.addText(bullets([
    'RQ1 — Năng lực nền: tiền huấn luyện từ đầu ở quy mô 60–63M khác gì so với mô hình đã tiền huấn luyện 135M–3B về độ trôi chảy và tuân thủ điều kiện.',
    'RQ2 — Can thiệp huấn luyện: ngân sách token, phân bố dữ liệu, vị trí LoRA và số chu kỳ ảnh hưởng ra sao đến chất lượng.',
    'RQ3 — Sử dụng điều kiện: mô hình thay đổi diễn biến theo điều kiện, hay chỉ tái tạo từ khóa quen thuộc.',
    'RQ4 — Kiểm soát lúc suy luận: validation, chuẩn hóa định dạng và rewrite đóng góp bao nhiêu so với năng lực của checkpoint.',
  ]), { x: 0.44, y: 4.2, w: 12.4, h: 2.6 });
  vline(s, 7.85, 1.75, 1.85);
  s.addText('Năm trường điều kiện', { x: 8.15, y: 1.75, w: 4.5, h: 0.35, fontFace: F, fontSize: 15, bold: true, color: C.teal });
  s.addText([
    { text: 'Nhân vật  ·  Bối cảnh', options: { breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Thử thách  ·  Kết quả', options: { breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Bài học (Moral)', options: {} },
  ], { x: 8.15, y: 2.2, w: 4.6, h: 1.4, fontFace: FL, fontSize: 19, color: C.ink, lineSpacingMultiple: 1.1 });
}

// ============================ 3. DỮ LIỆU & CÀI CẮM ĐIỀU KIỆN ============================
{
  const s = p.addSlide(); chrome(s, 'Dữ liệu và cách cài cắm điều kiện đầu vào');
  // hàng số liệu nổi bật
  stat(s, 0.44, 1.72, '≈ 3 triệu truyện', 'TF1-EN-3M (klusai/ds-tf1-en-3m): truyện ngụ ngôn tổng hợp có bài học (Nadas et al., 2025, arXiv 2504.20605).', C.blue, 4.0);
  stat(s, 4.66, 1.72, '5 trường điều kiện', 'Nhân vật · Bối cảnh · Thử thách · Kết quả · Bài học, đi kèm mỗi truyện.', C.blue, 4.0);
  stat(s, 8.88, 1.72, '934M token', 'Toàn bộ TF1 sau lọc: 2,34M truyện, ngữ cảnh 1.024, dùng pretrain E1 60M.', C.blue, 3.95);
  s.addShape(p.ShapeType.line, { x: 0.44, y: 3.15, w: 12.45, h: 0, line: { color: C.line, width: 1 } });
  // cột trái: cấu trúc bản ghi & quy mô tập con
  s.addText('Cấu trúc bản ghi và quy mô sử dụng', { x: 0.44, y: 3.3, w: 7, h: 0.35, fontFace: F, fontSize: 15, bold: true, color: C.teal });
  s.addText(bullets([
    'Mỗi bản ghi gồm một đề bài có cấu trúc (năm trường) và một truyện thiếu nhi tiếng Anh tương ứng, kèm một dòng bài học tường minh; nguồn công khai trên HuggingFace.',
    'Mỗi hướng lấy tập con và bộ tách từ riêng: E1 dùng toàn bộ TF1 sau lọc, E3 khoảng 50.000 mẫu, E4 khoảng 10.000 mẫu độ trôi chảy, E5 một nghìn mẫu đầu.',
    'Vì kích thước tập và bộ tách từ khác nhau nên perplexity không so trực tiếp được giữa các hướng; mọi so sánh liên nhóm đều dùng chung một giám khảo.',
  ]), { x: 0.44, y: 3.72, w: 7.2, h: 3.0 });
  vline(s, 7.95, 3.3, 3.5);
  // cột phải: định dạng cài cắm + kỹ thuật giữ đa dạng
  s.addText('Cài cắm điều kiện và giữ đa dạng', { x: 8.25, y: 3.3, w: 4.6, h: 0.35, fontFace: F, fontSize: 15, bold: true, color: C.teal });
  s.addText('<năm trường>  <|story|>  <truyện>  <|end|>',
    { x: 8.25, y: 3.72, w: 4.6, h: 0.5, fontFace: 'Courier New', fontSize: 12.5, color: C.ink, fill: { color: C.panel }, align: 'left', valign: 'middle', margin: 6 });
  s.addText(bullets([
    'Với hướng huấn luyện từ đầu (E1, E2), token phần điều kiện bị mask -100 nên loss chỉ tính trên truyện: mô hình học kể chuyện, không học thuộc mẫu đề.',
    'Hai giao diện điều kiện: E1, E3, E4, E5 dùng đủ năm trường; E2 dùng Nhân vật + Bài học theo hợp đồng V16.',
    'Giữ đa dạng: slot dropout che ngẫu nhiên từng trường, cap tỷ lệ khuôn mẫu chiếm ưu thế, và khử trùng lặp theo mã băm đề bài.',
  ]), { x: 8.25, y: 4.35, w: 4.6, h: 2.4 });
}

// ============================ 4. HƯỚNG GIẢI QUYẾT ============================
{
  const s = p.addSlide(); chrome(s, 'Hướng giải quyết: ba nhóm phương pháp trên cùng một tác vụ');
  const col = (x, tag, tagColor, head, body, sys) => {
    s.addShape(p.ShapeType.roundRect, { x, y: 1.85, w: 3.95, h: 4.5, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    s.addText(tag, { x: x + 0.25, y: 2.1, w: 3.5, h: 0.35, fontFace: F, fontSize: 13, bold: true, color: tagColor, charSpacing: 1 });
    s.addText(head, { x: x + 0.25, y: 2.5, w: 3.5, h: 0.8, fontFace: F, fontSize: 16.5, bold: true, color: C.ink, lineSpacingMultiple: 0.98 });
    s.addText(body, { x: x + 0.25, y: 3.4, w: 3.5, h: 2.1, fontFace: F, fontSize: 13, color: C.gray, lineSpacingMultiple: 1.05 });
    s.addText(sys, { x: x + 0.25, y: 5.75, w: 3.5, h: 0.5, fontFace: F, fontSize: 12.5, italic: true, color: tagColor });
  };
  col(0.44, 'NHÓM 1 · TIỀN HUẤN LUYỆN TỪ ĐẦU', C.teal, 'Xây năng lực nền từ khởi tạo ngẫu nhiên',
    'Thiết kế bộ tách từ, kiến trúc và ngân sách token; kiểm soát toàn bộ chuỗi xử lý để quy kết mỗi thay đổi chất lượng về đúng một nguyên nhân.', 'E1 · Llama 30M/60M      E2 · GPT-2 63M');
  col(4.69, 'NHÓM 2 · PEFT / QLoRA', C.orange, 'Thích nghi một mô hình đã tiền huấn luyện',
    'Giữ mô hình nền, chỉ học các cập nhật hạng thấp; khảo sát vị trí đặt adapter và ngân sách huấn luyện để đạt hiệu quả với chi phí tham số nhỏ.', 'E3 · SmolLM2-135M + LoRA      E5 · Llama 3.2 3B + QLoRA');
  col(8.94, 'NHÓM 3 · KIỂM SOÁT ĐẦU RA', C.green, 'Bảo đảm độ tin cậy tại thời điểm suy luận',
    'Kiểm tra định dạng, hậu xử lý và viết lại có điều kiện quanh một mô hình nền mạnh, nhằm nâng độ tin cậy của sản phẩm cuối.', 'E4 · Llama 3.2 3B + Repair');
  s.addText('Hạ tầng dùng chung: ứng dụng FastAPI/React với chế độ so sánh, phục vụ qua Ollama/MLX; đánh giá vòng chung trên 25 đề cố định, chấm mù bằng cùng một giám khảo Gemma.',
    { x: 0.44, y: 6.5, w: 12.4, h: 0.6, fontFace: F, fontSize: 12.5, italic: true, color: C.gray, align: 'center' });
}

// ============================ 5. BA LỚP BẰNG CHỨNG ============================
{
  const s = p.addSlide(); chrome(s, 'Ba lớp bằng chứng cho cùng một tác vụ');
  s.addText('Một truyện trôi chảy chưa chứng minh rằng các điều kiện đã chi phối diễn biến. Do đó chất lượng được báo cáo trên ba lớp độc lập, không thể thay thế lẫn nhau.',
    { x: 0.44, y: 1.75, w: 12.4, h: 0.7, fontFace: F, fontSize: 15, color: C.ink, lineSpacingMultiple: 1.05 });
  const lay = (x, tag, head, body) => {
    s.addShape(p.ShapeType.roundRect, { x, y: 2.7, w: 3.95, h: 3.4, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.blue, width: 1.25 } });
    s.addText(tag, { x: x + 0.25, y: 2.95, w: 3.5, h: 0.7, fontFace: F, fontSize: 17, bold: true, color: C.blue, lineSpacingMultiple: 0.98 });
    s.addText(head, { x: x + 0.25, y: 3.75, w: 3.5, h: 0.4, fontFace: F, fontSize: 12.5, bold: true, color: C.gray });
    s.addText(body, { x: x + 0.25, y: 4.2, w: 3.5, h: 1.7, fontFace: F, fontSize: 12.5, color: C.ink, lineSpacingMultiple: 1.05 });
  };
  lay(0.44, 'Độ giống với dữ liệu', 'Loss · PPL · fluency · Distinct', 'Mô hình có học được phong cách dữ liệu và tạo văn bản dễ đọc hay không. Đo chất lượng bề mặt.');
  lay(4.69, 'Mức độ dùng điều kiện', 'Coverage · trait→choice→outcome', 'Độ phủ các trường, chuỗi nhân quả, quan hệ truyện–bài học và mức nhạy khi thay đổi đúng một điều kiện.');
  lay(8.94, 'Độ tin cậy hệ thống', 'Validator · postprocess · repair', 'Kiểm tra và sửa định dạng bảo đảm đầu ra dùng được; các bước này không thay thế năng lực sinh của mô hình.');
  s.addText('Điểm tổng hợp chỉ có ý nghĩa khi ba lớp được báo cáo và diễn giải riêng.', { x: 0.44, y: 6.3, w: 12.4, h: 0.4, fontFace: F, fontSize: 13.5, italic: true, color: C.gray, align: 'center' });
}

// ======================================================================
// HÀM DỰNG 3 SLIDE MỖI E
// ======================================================================
// a = { type, arch:[...], rationale, config:[[k,v]...], train:{loss,optimizer,batch,hardware,reg}, flow }
function eArch(e, name, a) {
  const s = p.addSlide(); chrome(s, `${e} · ${name} — Kiến trúc và huấn luyện`, { eyebrow: `TINYSTORY-VN · ${e}` });
  // CỘT TRÁI — 1. Kiến trúc
  s.addText('1 · Kiến trúc', { x: 0.44, y: 1.72, w: 6.1, h: 0.32, fontFace: F, fontSize: 14, bold: true, color: C.teal });
  s.addText(a.type, { x: 0.44, y: 2.06, w: 6.15, h: 0.5, fontFace: F, fontSize: 13, bold: true, color: C.ink, lineSpacingMultiple: 0.98 });
  s.addText(a.arch.map((t, i) => ({ text: t, options: { bullet: { code: '2022', indent: 12 }, breakLine: true, paraSpaceAfter: 5, fontFace: F, fontSize: 11.5, color: C.ink } })),
    { x: 0.44, y: 2.6, w: 6.15, h: 1.55 });
  s.addText([{ text: 'Lý do chọn: ', options: { bold: true, color: C.teal } }, { text: a.rationale, options: { color: C.ink } }],
    { x: 0.44, y: 4.18, w: 6.15, h: 0.72, fontFace: F, fontSize: 11.5, lineSpacingMultiple: 1.0 });
  const crows = a.config.map(([k, v]) => [
    { text: k, options: { fontFace: F, fontSize: 11, color: C.gray, valign: 'middle' } },
    { text: v, options: { fontFace: F, fontSize: 11, bold: true, color: C.ink, align: 'right', valign: 'middle' } },
  ]);
  s.addTable(crows, { x: 0.44, y: 4.98, w: 6.15, colW: [2.7, 3.45], border: { type: 'solid', color: C.line, pt: 0.5 }, rowH: 0.26, valign: 'middle' });
  vline(s, 6.72, 1.72, 4.55);
  // CỘT PHẢI — 2. Thiết lập huấn luyện
  s.addText('2 · Thiết lập huấn luyện', { x: 6.95, y: 1.72, w: 5.95, h: 0.32, fontFace: F, fontSize: 14, bold: true, color: C.teal });
  const T = a.train;
  const trows = [['Hàm mất mát', T.loss], ['Optimizer', T.optimizer], ['Batch / epoch', T.batch], ['Phần cứng', T.hardware], ['Regularization', T.reg]].map(([k, v]) => [
    { text: k, options: { fontFace: F, fontSize: 11, bold: true, color: C.teal, valign: 'middle', fill: { color: C.panel } } },
    { text: v, options: { fontFace: F, fontSize: 10.5, color: C.ink, valign: 'middle' } },
  ]);
  s.addTable(trows, { x: 6.95, y: 2.12, w: 5.95, colW: [1.55, 4.4], border: { type: 'solid', color: C.line, pt: 0.5 }, rowH: 0.62, valign: 'middle', margin: [3, 5, 3, 5] });
  // DẢI DƯỚI — 3. Cách học và sinh token
  s.addShape(p.ShapeType.roundRect, { x: 0.44, y: 6.36, w: 12.0, h: 0.6, rectRadius: 0.05, fill: { color: C.panel }, line: { color: C.line, width: 0.75 } });
  s.addText([{ text: '3 · Cách học và sinh token    ', options: { bold: true, color: C.teal } }, { text: a.flow, options: { color: C.ink } }],
    { x: 0.62, y: 6.36, w: 11.7, h: 0.6, fontFace: F, fontSize: 11.5, valign: 'middle', lineSpacingMultiple: 0.98 });
}
function eProg(e, name, subtitle, steps, chartFile, chartCap) {
  const s = p.addSlide(); chrome(s, `${e} · ${name} — Tiến trình nghiên cứu`, { eyebrow: `TINYSTORY-VN · ${e}` });
  s.addText(subtitle, { x: 0.44, y: 1.72, w: 12.4, h: 0.45, fontFace: F, fontSize: 14.5, italic: true, color: C.gray, lineSpacingMultiple: 1.0 });
  // cột trái: các mốc (bảng)
  const rows = steps.map(([mile, note]) => [
    { text: mile, options: { fontFace: F, fontSize: 12, bold: true, color: C.teal, valign: 'middle' } },
    { text: note, options: { fontFace: F, fontSize: 11.5, color: C.ink, valign: 'middle' } },
  ]);
  s.addTable(rows, { x: 0.44, y: 2.35, w: 6.0, colW: [1.35, 4.65], border: { type: 'solid', color: C.line, pt: 0.5 }, rowH: 0.3, valign: 'middle' });
  // cột phải: biểu đồ tiến độ
  img(s, chartFile, { x: 6.75, y: 2.25, w: 6.2, h: 4.0 });
  s.addText(chartCap, { x: 6.75, y: 6.35, w: 6.2, h: 0.6, fontFace: F, fontSize: 11, italic: true, color: C.gray, align: 'center', lineSpacingMultiple: 1.0 });
}
function eResult(e, name, stats, analysis, chartFile, chartCap, limitation) {
  const s = p.addSlide(); chrome(s, `${e} · ${name} — Kết quả và phân tích`, { eyebrow: `TINYSTORY-VN · ${e}` });
  // trái: chart
  if (chartFile) { img(s, chartFile, { x: 0.44, y: 1.9, w: 6.3, h: 4.1 }); s.addText(chartCap, { x: 0.44, y: 6.05, w: 6.3, h: 0.7, fontFace: F, fontSize: 11, italic: true, color: C.gray, align: 'center', lineSpacingMultiple: 1.0 }); }
  vline(s, 7.0, 1.85, 4.7);
  let y = 1.9;
  stats.forEach(([big, desc, color]) => { stat(s, 7.3, y, big, desc, color || C.blue, 5.4); y += 1.35; });
  s.addText('Phân tích và giới hạn', { x: 7.3, y: y + 0.02, w: 5.4, h: 0.35, fontFace: F, fontSize: 14, bold: true, color: C.teal });
  s.addText(analysis, { x: 7.3, y: y + 0.42, w: 5.4, h: 1.0, fontFace: F, fontSize: 12.5, color: C.ink, lineSpacingMultiple: 1.03 });
  if (limitation) s.addText(limitation, { x: 0.44, y: 6.75, w: 6.3, h: 0.5, fontFace: F, fontSize: 11.5, italic: true, color: C.gray, lineSpacingMultiple: 1.0 });
}

// SƠ ĐỒ KHỐI các bước phương pháp (serpentine 3 cột), dùng chung cho E2-E5
const PHASE = {
  'nền-tảng': { c: C.teal, label: 'Nền tảng & thiết lập' },
  'can-thiệp': { c: C.blue, label: 'Can thiệp & huấn luyện' },
  'phát-hiện': { c: C.orange, label: 'Đo lường & phát hiện' },
  'kết-luận': { c: C.green, label: 'Kết luận & kết quả' },
};
function mFlow(titleFull, eyebrow, subtitle, nodes) {
  const s = p.addSlide(); chrome(s, titleFull, { eyebrow });
  s.addText(subtitle, { x: 0.44, y: 1.68, w: 12.45, h: 0.45, fontFace: F, fontSize: 12.5, italic: true, color: C.gray, lineSpacingMultiple: 1.0 });
  const cX = [0.44, 4.74, 9.04], rY = [2.35, 3.98, 5.61], BW = 3.85, BH = 1.32;
  const place = (i) => { const row = Math.floor(i / 3), k = i % 3; const col = (row % 2 === 0) ? k : (2 - k); return { row, col, x: cX[col], y: rY[row] }; };
  nodes.forEach((n, i) => {
    const pos = place(i), col = PHASE[n.phase].c;
    s.addShape(p.ShapeType.roundRect, { x: pos.x, y: pos.y, w: BW, h: BH, rectRadius: 0.07, fill: { color: C.white }, line: { color: col, width: 1.5 } });
    s.addShape(p.ShapeType.roundRect, { x: pos.x, y: pos.y, w: 0.12, h: BH, rectRadius: 0.02, fill: { color: col }, line: { type: 'none' } });
    s.addText([
      { text: n.id + '  ', options: { bold: true, color: col } },
      { text: n.title, options: { bold: true, color: C.ink } },
    ], { x: pos.x + 0.24, y: pos.y + 0.11, w: BW - 0.42, h: 0.5, fontFace: F, fontSize: 11.5, lineSpacingMultiple: 0.95 });
    s.addText(n.cap, { x: pos.x + 0.24, y: pos.y + 0.62, w: BW - 0.42, h: 0.64, fontFace: F, fontSize: 9.5, color: C.gray, lineSpacingMultiple: 0.97 });
  });
  const AR = { color: C.mute, width: 2, endArrowType: 'triangle' };
  const ARL = { color: C.mute, width: 2, beginArrowType: 'triangle' };
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = place(i), b = place(i + 1);
    if (a.row === b.row) {
      const midY = a.y + BH / 2;
      if (b.col > a.col) s.addShape(p.ShapeType.line, { x: a.x + BW + 0.03, y: midY, w: 0.33, h: 0, line: AR });
      else s.addShape(p.ShapeType.line, { x: b.x + BW + 0.03, y: midY, w: 0.33, h: 0, line: ARL });
    } else {
      s.addShape(p.ShapeType.line, { x: a.x + BW / 2, y: a.y + BH + 0.02, w: 0, h: (b.y - a.y - BH) - 0.04, line: AR });
    }
  }
  const order = ['nền-tảng', 'can-thiệp', 'phát-hiện', 'kết-luận'];
  const present = order.filter(ph => nodes.some(n => n.phase === ph));
  const leg = [];
  present.forEach((ph, idx) => {
    leg.push({ text: '■ ', options: { color: PHASE[ph].c } });
    leg.push({ text: PHASE[ph].label + (idx < present.length - 1 ? '      ' : ''), options: { color: C.gray } });
  });
  s.addText(leg, { x: 0.44, y: 7.02, w: 12.45, h: 0.3, fontFace: F, fontSize: 10, align: 'center' });
}

// ---------------- E1 ----------------
eArch('E1', 'Llama 30M/60M từ đầu', {
  type: 'Transformer decoder-only kiểu Llama, huấn luyện từ khởi tạo ngẫu nhiên (from-scratch).',
  arch: [
    'Bộ ba hiện đại RoPE, RMSNorm và SwiGLU; embedding vào và ra dùng chung (tied); attention GQA giảm bộ nhớ đệm khi sinh.',
    'Bộ tách từ BPE 12k tự huấn luyện; hai quy mô 30M và 60M cùng họ để cô lập ảnh hưởng của dung lượng.',
  ],
  rationale: 'Các thành phần Llama hiệu quả ở quy mô nhỏ và vocab 12k giữ bảng embedding gọn (~17% ngân sách), dành tham số cho các khối xử lý.',
  config: [['Tham số', '36,6M / 59,56M'], ['Khối / hidden', '8 / 512→768'], ['Attention', 'GQA 8→12 q / 2→4 kv'], ['FFN', 'SwiGLU 2.048'], ['Vocab / ngữ cảnh', '12k / 512→1024']],
  train: {
    loss: 'Cross-entropy next-token, mask phần điều kiện (-100), chỉ tính trên token truyện.',
    optimizer: 'AdamW, learning rate đỉnh 3e-3 lịch WSD (warmup, stable, decay), có weight decay và gradient clip.',
    batch: 'Batch hiệu dụng 128 (gradient accumulation); Phase 1 ~600M token (4 epoch); 60M dùng 934M token, 10.000 bước không lặp.',
    hardware: 'Colab T4, checkpoint tự khôi phục qua nhiều phiên khi runtime bị thu hồi.',
    reg: 'Weight decay, gradient clip, slot dropout; huấn luyện fp16.',
  },
  flow: 'Học: gradient chỉ dồn vào token truyện sau <|story|>; Sinh: nối token tự hồi quy tới khi gặp <|end|>.',
});
eProg('E1', 'Llama 30M/60M từ đầu',
  'Bốn giai đoạn: chẩn đoán under-training, cấp đủ token, can thiệp phân bố dữ liệu, rồi mở rộng quy mô để kiểm chứng.',
  [
    ['v1', 'Cố ý giới hạn dữ liệu (150k) để chẩn đoán; judge 2,50 — under-training.'],
    ['Phase 1', 'Tăng token (400k × 4 epoch); loss 1,447, judge 6,00. Đúng chẩn đoán.'],
    ['Phase 2', 'Can thiệp phân bố dữ liệu: cap khuôn "wise old owl" 90%→23%; judge 7,00, loss 1,278.'],
    ['Hậu HL', 'DPO / SFT-best / RAFT / GRPO trung tính; distillation âm; best-of-N +0,8.'],
    ['60M', 'Mở rộng hidden/ngữ cảnh + toàn bộ TF1 (934M token); loss 1,058, PPL 2,87.'],
  ],
  fig('trieulh/report/figures/01_loss_curve.png'),
  'Loss huấn luyện qua hai giai đoạn: giảm nhanh ở Phase 1, tiếp tục giảm sau khi nối Phase 2 (corpus v2); loss cuối 1,278, thấp hơn rõ so với baseline v1 1,8.');
// --- slide bổ sung: chín bước phương pháp M1 -> M9 (SƠ ĐỒ KHỐI) ---
{
  const s = p.addSlide(); chrome(s, 'E1 · Llama 30M/60M từ đầu — Chín bước phương pháp (M1 đến M9)', { eyebrow: 'TINYSTORY-VN · E1' });
  s.addText('Mạch xuyên suốt: chẩn đoán bệnh, chữa bằng dữ liệu, đo cho chuẩn, thử các đường tắt hậu huấn luyện (đều thất bại), rồi kết luận phải đầu tư pretraining và mở rộng để xác nhận.',
    { x: 0.44, y: 1.68, w: 12.45, h: 0.45, fontFace: F, fontSize: 12.5, italic: true, color: C.gray, lineSpacingMultiple: 1.0 });
  // toạ độ lưới 3x3
  const cX = [0.44, 4.74, 9.04], rY = [2.35, 3.98, 5.61];
  const BW = 3.85, BH = 1.32;
  const node = (col, row, num, numColor, title, cap) => {
    const x = cX[col], y = rY[row];
    s.addShape(p.ShapeType.roundRect, { x, y, w: BW, h: BH, rectRadius: 0.07, fill: { color: C.white }, line: { color: numColor, width: 1.5 } });
    s.addShape(p.ShapeType.roundRect, { x, y, w: 0.12, h: BH, rectRadius: 0.02, fill: { color: numColor }, line: { type: 'none' } }); // dải màu trái
    s.addText([
      { text: num + '  ', options: { bold: true, color: numColor } },
      { text: title, options: { bold: true, color: C.ink } },
    ], { x: x + 0.24, y: y + 0.13, w: BW - 0.42, h: 0.5, fontFace: F, fontSize: 12, lineSpacingMultiple: 0.95 });
    s.addText(cap, { x: x + 0.24, y: y + 0.66, w: BW - 0.42, h: 0.58, fontFace: F, fontSize: 10, color: C.gray, lineSpacingMultiple: 0.98 });
  };
  const AR = { color: C.mute, width: 2, endArrowType: 'triangle' };
  const ARL = { color: C.mute, width: 2, beginArrowType: 'triangle' };
  const arrowR = (xs, y) => s.addShape(p.ShapeType.line, { x: xs, y, w: 0.33, h: 0, line: AR });
  const arrowL = (xs, y) => s.addShape(p.ShapeType.line, { x: xs, y, w: 0.33, h: 0, line: ARL });
  const arrowD = (x, ys) => s.addShape(p.ShapeType.line, { x, y: ys, w: 0, h: 0.31, line: AR });

  // Hàng 1 (nền tảng, teal): M1 -> M2 -> M3
  node(0, 0, 'M1', C.teal, 'Tokenizer BPE 12k', 'Bảng token riêng 12k, embedding gọn (~17% ngân sách 30M).');
  node(1, 0, 'M2', C.teal, 'Dữ liệu 5 trường', 'Ghép đề + truyện, mask -100, slot dropout, khử trùng lặp.');
  node(2, 0, 'M3', C.teal, 'Kiến trúc Llama 30M', 'Decoder 8 khối, GQA, RoPE/RMSNorm/SwiGLU, 36,6M tham số.');
  // Hàng 2 (chẩn đoán & chữa): M4 (teal) <- ngược lại M6, M5
  node(2, 1, 'M4', C.teal, 'Vòng lặp huấn luyện', 'AdamW, learning rate WSD, fp16, checkpoint tự khôi phục.');
  node(1, 1, 'M5', C.blue, 'Chẩn đoán under-training', 'Thiếu token: judge 2,5; đủ ~600M token: judge 6,0.');
  node(0, 1, 'M6', C.blue, 'Can thiệp phân bố dữ liệu', 'Hạ khuôn "owl" 90%→23%; judge 7,0, loss 1,278.');
  // Hàng 3: M7 (đo lường, gray) -> M8 (đường tắt, cam) -> M9 (kết luận, xanh lá)
  node(0, 2, 'M7', C.gray, 'Đo lường & nhiễu judge', 'PPL held-out; nhiễu ±0,4 → n=45 seed bắt cặp, t-test.');
  node(1, 2, 'M8', C.orange, 'Hậu huấn luyện (5 cách)', 'DPO/RAFT/GRPO/RM/distill trung tính; best-of-N +0,8.');
  node(2, 2, 'M9', C.green, 'Mở rộng 60M kiểm chứng', '934M token; judge 7,94→8,96 (+1,017, thắng 36/45).');

  // mũi tên nối
  const m1 = rY[0] + BH / 2, m2 = rY[1] + BH / 2, m3 = rY[2] + BH / 2;
  arrowR(cX[0] + BW + 0.03, m1); arrowR(cX[1] + BW + 0.03, m1);          // M1->M2->M3
  arrowD(cX[2] + BW / 2, rY[0] + BH + 0.02);                              // M3 xuống M4
  arrowL(cX[1] + BW + 0.03, m2); arrowL(cX[0] + BW + 0.03, m2);          // M4<-M5<-M6 (arrow points left)
  arrowD(cX[0] + BW / 2, rY[1] + BH + 0.02);                              // M6 xuống M7
  arrowR(cX[0] + BW + 0.03, m3); arrowR(cX[1] + BW + 0.03, m3);          // M7->M8->M9

  // chú giải giai đoạn
  s.addText([
    { text: '■ ', options: { color: C.teal } }, { text: 'Xây nền tảng    ', options: { color: C.gray } },
    { text: '■ ', options: { color: C.blue } }, { text: 'Chẩn đoán & chữa bằng dữ liệu    ', options: { color: C.gray } },
    { text: '■ ', options: { color: C.gray } }, { text: 'Đo lường    ', options: { color: C.gray } },
    { text: '■ ', options: { color: C.orange } }, { text: 'Đường tắt (thất bại)    ', options: { color: C.gray } },
    { text: '■ ', options: { color: C.green } }, { text: 'Kết luận & mở rộng', options: { color: C.gray } },
  ], { x: 0.44, y: 7.02, w: 12.45, h: 0.3, fontFace: F, fontSize: 10, align: 'center' });
}
eResult('E1', 'Llama 30M/60M từ đầu',
  [['+1,017', '60M so với 30M Phase 2 (n=45, seed bắt cặp); cao hơn ở 36/45 đề, t = 6,53.', C.blue],
   ['8,55', 'Best-of-3 từ trung bình một lần sinh 7,72: mô hình đã chứa mẫu tốt nhưng phân bố chưa ổn định.', C.blue],
   ['3,30', 'Điểm vòng chung: runner không truyền token <|story|>, tạo đầu vào lệch hợp đồng prompt.', C.gray]],
  'Trong phạm vi E1, mở rộng quy mô tiền huấn luyện hiệu quả hơn mọi cấu hình hậu huấn luyện đã thử. Chênh lệch giữa 8,96 nội bộ và 3,30 vòng chung phản ánh độ nhạy với định dạng prompt, không phải năng lực kém.',
  fig('trieulh/report/figures/19_headtohead_progression.png'),
  'Đối đầu các checkpoint E1 ở hai chế độ sinh và hai giám khảo.',
  'Giới hạn: 60M thay đổi đồng thời tham số, ngữ cảnh và ngân sách token — chưa cô lập riêng ảnh hưởng của dung lượng.');

// ---------------- E2 ----------------
eArch('E2', 'GPT-2 63M từ đầu', {
  type: 'Transformer decoder-only kiểu GPT-2 (GPT2LMHeadModel), 7 khối, huấn luyện từ đầu.',
  arch: [
    'Khối LayerNorm → multi-head attention → residual → LayerNorm → GELU MLP → residual, với bảng vị trí học được và weight tying vào/ra.',
    'Bộ tách từ Metaspace BPE 16.384; điều kiện đóng thẻ character/moral/story, loss phần prompt bị mask.',
  ],
  rationale: 'Chọn GPT-2 decoder-only train từ khởi tạo ngẫu nhiên để kiểm tra chất lượng biểu diễn token và khả năng dùng character + moral làm điều kiện.',
  config: [['Tham số', '62,99M'], ['Khối / hidden', '7 / 768'], ['Attention', 'MHA 12 head, 64 chiều'], ['FFN', 'GELU-new 3.072'], ['Tokenizer', 'Metaspace BPE 16k']],
  train: {
    loss: 'Cross-entropy next-token, mask phần prompt (label -100), chỉ tính loss trên thân truyện.',
    optimizer: 'Learning rate ba pha 5e-4 → 1e-4 → 3e-6, weight decay 0,1, lịch warmup-decay (tên optimizer không nêu trong báo cáo).',
    batch: 'Tiền HL 2 epoch/15.625 bước; điều kiện hóa 1.611 bước; pha chỉ-nhân-quả 3 epoch/624 bước.',
    hardware: 'Colab, một GPU A100; ba pha V16 chạy khoảng 100 phút.',
    reg: 'Mask prompt (-100), weight decay 0,1; dropout và gradient clip không nêu số.',
  },
  flow: 'Học: nhận gradient trên token truyện sau <story>; Sinh: nối token truyện cho tới khi gặp </story>.',
});
eProg('E2', 'GPT-2 63M từ đầu',
  'Chuỗi thí nghiệm nối tiếp theo giả thuyết: sửa tokenizer, bám trường đầu vào, rồi nhiều hướng nhằm tạo lập kế hoạch nhân quả.',
  [
    ['V1–V3', 'Sửa ranh giới từ; khớp đủ hai trường 3%→55%; kết thúc sạch 0%→100%.'],
    ['V7–V10', 'Chưng cất plan+story, trộn replay nhân quả; giữ trôi chảy, causal-pass ~2%.'],
    ['V11–V13', 'Thẻ moral_class, tăng layer 63M→98M, DPO; causal-pass không đổi.'],
    ['V15', 'Loss phụ liên kết story–moral: matching 75,4% nhưng không thành khả năng sinh.'],
    ['V16', 'Tăng tiền huấn luyện 200k→500k: fluency tăng, liên kết nhân quả vẫn không tăng.'],
  ],
  fig('figures/tracks/e2_v10_causal_replay.png'),
  'V10 — replay nhân quả: loss validation nhân quả giảm 3,404→2,968 nhưng causal-pass giữ ở mức ~2%.');
mFlow('E2 · GPT-2 63M từ đầu — Các bước phương pháp (V1 đến V16)', 'TINYSTORY-VN · E2',
  'Mạch nghiên cứu: sửa biểu diễn token, bám trường đầu vào, rồi nhiều nỗ lực tạo lập kế hoạch nhân quả; kết luận loss thấp và matching cao vẫn không đủ để điều kiện chi phối diễn biến.',
  [
    { id: 'V1-V2', title: 'Thiết kế lại tokenizer', cap: 'Metaspace BPE 16.384 xóa lỗi vỡ từ; Distinct-1 +33%, Self-BLEU giảm 64%.', phase: 'nền-tảng' },
    { id: 'V3', title: 'Điều kiện hóa có mask', cap: 'Khớp cả hai trường 3% lên 55%, kết thúc </story> 0% lên 100%, fluency 6,21.', phase: 'nền-tảng' },
    { id: 'Chẩn đoán', title: 'Phát hiện lỗ hổng nhân quả', cap: 'V3 đúng nhân vật 96% nhưng trait chi phối lựa chọn 1%, plot suy ra bài học 0%.', phase: 'phát-hiện' },
    { id: 'V4-V8', title: 'Chưng cất teacher', cap: 'Dữ liệu bổ sung không cải thiện; V8 chưng cất teacher chỉ fluency 4,65, causal 0%.', phase: 'can-thiệp' },
    { id: 'V9-V11', title: 'Replay và thẻ lớp', cap: 'Replay giữ trôi chảy, causal-pass chỉ 2%; thẻ moral_class chọn kiểu truyện.', phase: 'can-thiệp' },
    { id: 'V12-V13', title: 'Tăng quy mô và DPO', cap: '63M lên 98M chỉ +1% causal (CI chạm 0); DPO đạt 54,4%, gần mức ngẫu nhiên.', phase: 'can-thiệp' },
    { id: 'V14-V15', title: 'Ép nhân quả và loss phụ', cap: '8 epoch causal làm fluency giảm; V15 matching 75,4% không thành khả năng sinh.', phase: 'can-thiệp' },
    { id: 'V16', title: 'Tăng tiền huấn luyện', cap: 'Tăng 200k lên 500k truyện; fluency 5,90 lên 6,85 nhưng causal-pass vẫn 0%.', phase: 'can-thiệp' },
    { id: 'Kết luận', title: 'Phát hiện cốt lõi', cap: 'Loss thấp, bám đầu vào cao, matching tốt vẫn không chi phối nhân quả; judge 3,18/10.', phase: 'kết-luận' },
  ]);
eResult('E2', 'GPT-2 63M từ đầu',
  [['3% → 55%', 'Tỷ lệ khớp đủ hai trường đầu vào sau khi sửa tokenizer và masking (V2→V3).', C.blue],
   ['0–5%', 'Causal-pass giữ nguyên qua mọi can thiệp: tăng depth, DPO và nhiều epoch đều không cải thiện.', C.gray],
   ['3,18', 'Điểm vòng chung; nhắc đúng nhân vật và bài học nhưng chưa lập kế hoạch nhân quả khi sinh.', C.gray]],
  'Cải thiện bộ tách từ và tăng token nâng chất lượng ngôn ngữ và độ bám trường, nhưng khả năng để điều kiện chi phối diễn biến không xuất hiện ở quy mô 63M. Loss và matching không chứng minh được lập kế hoạch khi sinh.',
  fig('figures/tracks/e2_v14_causal_epochs.png'),
  'V14 — tám epoch chỉ dùng dữ liệu nhân quả: fluency giảm, causal-pass vẫn 0–5%.',
  'Giới hạn: dữ liệu nhân quả bổ sung nhỏ; mỗi biến thể một seed.');

// ---------------- E3 ----------------
eArch('E3', 'SmolLM2-135M + LoRA', {
  type: 'SmolLM2-135M decoder-only kiểu Llama (30 tầng) đóng băng, chỉ học adapter LoRA hạng thấp.',
  arch: [
    'Giữ nguyên trọng số nền đã pretrain, chỉ chèn ma trận low-rank B·A vào các projection; ba nhánh A/B/C khác vị trí đặt adapter.',
    'SmolLM2 tách riêng q, k, v, o, gate, up, down nên khảo sát được phủ tầng và mở rộng sang MLP.',
  ],
  rationale: 'Kế thừa prior đã pretrain thay vì train từ đầu; thiết kế single-factor cho phép quy chênh lệch kết quả về đúng vị trí adapter.',
  config: [['Nền', '134,5M (đóng băng)'], ['Khối / hidden', '30 / 576'], ['Attention', 'GQA 9 q / 3 kv'], ['Adapter (C)', '4,88M (3,5%)'], ['Vocab / ngữ cảnh', '49k / 512']],
  train: {
    loss: 'Cross-entropy sinh có điều kiện, completion-only, chỉ tính trên token truyện (mask ngữ cảnh -100).',
    optimizer: 'AdamW, LR 2e-4 cosine, warmup 3%, bf16; LoRA r=16, alpha=32, dropout 0,05.',
    batch: 'Batch hiệu dụng 32 (16 × grad accum 2), 2 epoch (~3.125 bước/nhánh), tập con 50.000 truyện.',
    hardware: 'Một GPU Colab L4; vài phút mỗi nhánh (thời gian chi tiết không nêu).',
    reg: 'LoRA dropout 0,05; đóng băng toàn bộ nền; cosine với 3% warmup.',
  },
  flow: 'Học: chỉ tích low-rank B·A cập nhật trên token truyện; Sinh: dự đoán tuần tự từng token theo điều kiện system + đề.',
});
eProg('E3', 'SmolLM2-135M + LoRA',
  'Bốn cấu hình trên cùng nền, cùng dữ liệu và cùng ngân sách; chỉ đổi vị trí đặt adapter qua ba nhánh so với mô hình nền.',
  [
    ['Nền', 'Không adapter; PPL 9,52, điểm nội bộ 5,73 — mốc tham chiếu.'],
    ['A · q/v toàn tầng', '≈0,92M tham số; PPL 4,82, điểm 6,70; dẫn đầu sáng tạo và độ dễ đọc.'],
    ['B · q/v 1/3 tầng cuối', '≈0,31M; PPL 5,46, điểm 5,94; adherence 4,78 < nền 5,08.'],
    ['C · mọi lớp tuyến tính', '≈4,88M; PPL 3,84, điểm 6,87; dẫn đầu ngữ pháp, bài học, tuân thủ, overall.'],
  ],
  fig('figures/tracks/e3_train_dynamics.png'),
  'Động lực huấn luyện: cross-entropy thật trên token truyện; thứ hạng hội tụ C < A < B trùng thứ hạng perplexity held-out.');
mFlow('E3 · SmolLM2-135M + LoRA — Các bước phương pháp', 'TINYSTORY-VN · E3',
  'Mạch nghiên cứu: đóng băng mô hình nền, thiết kế ablation một biến về vị trí adapter, huấn luyện ba nhánh cùng điều kiện rồi so sánh perplexity và giám khảo nội bộ để tìm vị trí tối ưu.',
  [
    { id: 'B1', title: 'Chọn nền đóng băng', cap: 'SmolLM2-135M, 30 tầng, đóng băng toàn bộ trọng số; chỉ học adapter.', phase: 'nền-tảng' },
    { id: 'B2', title: 'Định dạng sinh có điều kiện', cap: 'system + đề 5 trường → truyện, loss chỉ trên token truyện (mask -100), seq 512.', phase: 'nền-tảng' },
    { id: 'B3', title: 'Thiết kế ablation một biến', cap: 'Tách trục độ sâu (A vs B) và độ rộng module (A vs C); giữ r=16, α=32, dropout 0,05.', phase: 'nền-tảng' },
    { id: 'B4', title: 'Thiết lập baseline nền', cap: 'Nền không adapter làm mốc: PPL 9,52 và Flesch -66, gần như không đọc được.', phase: 'nền-tảng' },
    { id: 'B5', title: 'Huấn luyện ba nhánh A/B/C', cap: 'Cùng 50k truyện, 2 epoch; adapter 0,92M (A) / 0,31M (B) / 4,88M (C, tức 3,5%).', phase: 'can-thiệp' },
    { id: 'B6', title: 'Đo perplexity held-out', cap: 'Trên 500 dòng cố định: C 3,84 < A 4,82 < B 5,46, thấp hơn nền 9,52 khoảng 60%.', phase: 'phát-hiện' },
    { id: 'B7', title: 'Kiểm chứng judge nội bộ', cap: 'Qwen2.5-7B, 4 trục, n=50/nhánh: C 6,87 > A 6,70 > B 5,94 > nền 5,73, đúng thứ hạng PPL.', phase: 'phát-hiện' },
    { id: 'B8', title: 'Kết luận vị trí tối ưu', cap: 'Độ rộng module lấn át độ sâu tầng; C all-linear tốt nhất, B adherence 4,78 tụt dưới nền.', phase: 'kết-luận' },
    { id: 'B9', title: 'Hạn chế và mở rộng', cap: 'Một seed không CI, một judge chưa đo nhiễu, thiếu ô all-linear × 1/3 tầng cuối.', phase: 'kết-luận' },
  ]);
eResult('E3', 'SmolLM2-135M + LoRA',
  [['3,84', 'Perplexity held-out của nhánh C — thấp nhất trong bốn cấu hình.', C.blue],
   ['C > A > B', 'Cùng thứ tự trên cả perplexity lẫn điểm giám khảo nội bộ (Qwen2.5-7B, n=50/nhánh).', C.blue],
   ['2,81', 'Điểm vòng chung; runner thiếu system message khiến truyện đổi nhân vật hoặc kết sớm.', C.gray]],
  'Phủ toàn bộ tầng tốt hơn chỉ đặt ở tầng cuối, và mở rộng adapter sang MLP là mức tăng lớn nhất — với chỉ 3,5% tham số được huấn luyện. Đây là bằng chứng phản bác cách hiểu "fine-tune luôn không hiệu quả".',
  fig('figures/tracks/e3_lora_ablation.png'),
  'Thí nghiệm loại trừ vị trí LoRA: perplexity (trái) và điểm giám khảo nội bộ (phải); nhánh C dẫn đầu cả hai.',
  'Giới hạn: thiếu cấu hình all-linear × 1/3 tầng cuối; mỗi nhánh chỉ một seed.');

// ---------------- E4 ----------------
eArch('E4', 'Llama 3.2 3B + Repair', {
  type: 'Llama 3.2 3B Instruct (Q4_K_M) + pipeline kiểm soát đầu ra (sinh, validate, repair).',
  arch: [
    'Hệ đại diện bao quanh mô hình nền: sinh một lần, validator soát hình thức và độ dài, chỉ repair khi lỗi nghiêm trọng, rồi chuẩn hóa đúng một dòng Moral.',
    'Không tạo checkpoint fine-tune mới; các biến thể SFT/LoRA chỉ dùng để so sánh.',
  ],
  rationale: 'Ba hướng fine-tune đều bất lợi (SFT giảm trôi chảy, LoRA không sửa được bài học); kiểm soát tại suy luận tăng độ tin cậy mà giữ năng lực kể chuyện của nền 3B.',
  config: [['Nền', '3,21B (Q4_K_M)'], ['Khối / hidden', '28 / 3.072'], ['Attention', 'GQA 24 q / 8 kv'], ['FFN', 'SwiGLU 8.192'], ['Ngữ cảnh chạy', '2.048']],
  train: {
    loss: 'Hệ đại diện không huấn luyện mới; biến thể SFT/QLoRA không nêu hàm mất mát.',
    optimizer: 'Chỉ suy luận; biến thể Failure-LoRA: QLoRA 4-bit r=16/α=16, LR 1e-4, warmup 5%, fp16.',
    batch: 'Failure-LoRA: batch hiệu dụng 8, 3 epoch, 300 mẫu (270/30); Fluency-SFT 9.000/1.000.',
    hardware: 'Suy luận qua Ollama, GGUF Q4_K_M, ngữ cảnh 2.048; latency trung bình ~81,6 giây.',
    reg: 'Validator hình thức/độ dài, repair khi lỗi nặng, chuẩn hóa một dòng Moral, seed 5410.',
  },
  flow: 'Sinh assistant tối đa 400 token; validator soát, nếu lỗi nặng sinh lại tối đa 500 token rồi chuẩn hóa một dòng Moral.',
});
eProg('E4', 'Llama 3.2 3B + Repair',
  'Đánh giá nhân lực trên năm cấu hình, và bóc tách đóng góp của bước sửa lỗi so với đầu ra thô của mô hình nền.',
  [
    ['Base + Repair', 'Điểm nhân lực 4,32/5 — cao nhất trong các cấu hình E4.'],
    ['Strict + post', 'Prompt nghiêm ngặt kèm hậu xử lý: 4,20/5.'],
    ['Fluency-SFT', 'SFT 10k mẫu độ trôi chảy: 3,84/5.'],
    ['Repair (bóc tách)', 'Bài học đúng nguyên văn 20%→100%; nhất quán yêu cầu 9,00→9,64.'],
    ['Nhân quả', 'trait→choice 92%, choice→outcome 100% — repair không làm thay đổi.'],
  ],
  fig('figures/tracks/e4_human_eval.png'),
  'Đánh giá nhân lực năm cấu hình E4: Base + Repair đạt điểm cao nhất về độ tin cậy tổng thể.');
mFlow('E4 · Llama 3.2 3B + Repair — Các bước phương pháp', 'TINYSTORY-VN · E4',
  'Mạch nghiên cứu: thử fine-tune rồi chuyển sang kiểm soát đầu ra (prompt, hậu xử lý, repair), đánh giá nhân lực bảy cấu hình và bóc tách riêng đóng góp của bước repair.',
  [
    { id: 'B1', title: 'Chọn mô hình nền 3B', cap: 'Llama 3.2 3B Instruct Q4_K_M, 3,21 tỷ tham số, ngữ cảnh runner 2.048 token.', phase: 'nền-tảng' },
    { id: 'B2', title: 'Thử fine-tune SFT/LoRA', cap: 'SFT-clean-3k và Failure-LoRA 300 mẫu đều âm, không sửa được dòng bài học.', phase: 'nền-tảng' },
    { id: 'B3', title: 'Prompt nghiêm ngặt', cap: 'Exact-character lên 0,92, kết thúc sạch 0,96, nhưng bài học chưa đủ tin cậy.', phase: 'can-thiệp' },
    { id: 'B4', title: 'Hậu xử lý chuẩn hóa Moral', cap: 'Chuẩn hóa đúng một dòng Moral; exact-moral và clean-ending lên 1,00.', phase: 'can-thiệp' },
    { id: 'B5', title: 'Pipeline Base + Repair', cap: 'Sinh, validate rồi repair; chỉ gọi lại mô hình khi validator báo lỗi nghiêm trọng.', phase: 'can-thiệp' },
    { id: 'B6', title: 'Kiểm chứng Fluency-SFT 10k', cap: 'SFT 10.000 mẫu đã lọc vẫn cho Moral rỗng 0,76 và clean-ending 0,12, không cải thiện.', phase: 'phát-hiện' },
    { id: 'B7', title: 'Đánh giá nhân lực 7 cấu hình', cap: 'Base + Repair đạt 4,32/5, đứng đầu về bám đề, cấu trúc, bài học và an toàn.', phase: 'phát-hiện' },
    { id: 'B8', title: 'Bóc tách đóng góp repair', cap: 'Bài học nguyên văn 20% lên 100%, nhất quán 9,00 lên 9,64; nhân quả giữ 92% và 100%.', phase: 'phát-hiện' },
    { id: 'B9', title: 'Kết luận toàn hệ thống', cap: 'Vòng chung 9,20/10, đứng đầu năm hướng; 5/25 đề phải gọi mô hình lần hai.', phase: 'kết-luận' },
  ]);
eResult('E4', 'Llama 3.2 3B + Repair',
  [['9,20', 'Điểm vòng chung — cao nhất trong năm hệ thống đại diện.', C.green],
   ['20% → 100%', 'Tỷ lệ bài học đúng nguyên văn sau bước repair, không đổi hai liên kết nhân quả.', C.blue],
   ['5 / 25', 'Số đề mà pipeline phải gọi mô hình lần hai; chi phí của độ tin cậy.', C.gray]],
  'Kiểm soát khi sinh hiệu quả hơn các lần fine-tune đã thử. Quan hệ nhân quả giữa điều kiện và diễn biến đã có sẵn trong đầu ra thô của mô hình nền 3B; repair chủ yếu sửa hợp đồng đầu ra và các sai lệch cục bộ.',
  ASSET + '/automatic_reliability.png',
  'Độ tin cậy tự động: tỷ lệ đạt các ràng buộc định dạng trước và sau bước repair.',
  'Giới hạn: kho mã không lưu trainer_state của các lần SFT nên chưa tái lập được đường huấn luyện.');

// ---------------- E5 ----------------
eArch('E5', 'Llama 3.2 3B + QLoRA', {
  type: 'Llama 3.2 3B Instruct decoder-only + QLoRA 4-bit (adapter r=16), merge rồi xuất GGUF Q4_K_M.',
  arch: [
    'Nạp nền ở 4-bit qua Unsloth, chỉ học adapter LoRA trên toàn bộ attention và MLP (24,31M tham số, 0,75%).',
    'Sau huấn luyện merge adapter vào nền và lượng tử hóa GGUF Q4_K_M (~2,0GB) để phục vụ Ollama.',
  ],
  rationale: 'QLoRA nạp nền 4-bit và chỉ huấn luyện 0,75% tham số, cho phép fine-tune và chạy cục bộ Llama 3B trong ngân sách GPU phổ thông.',
  config: [['Nền (sau merge)', '3,21B'], ['Tham số LoRA', '24,31M (0,75%)'], ['Khối / hidden', '28 / 3.072'], ['Attention', 'GQA 24 q / 8 kv'], ['Artifact', 'GGUF Q4_K_M']],
  train: {
    loss: 'Cross-entropy token kế tiếp, áp trên toàn bộ chuỗi chat đã đóng gói (không riêng token truyện).',
    optimizer: 'AdamW 8-bit, LR đỉnh 2e-4 giảm tuyến tính, warmup 5 bước, weight decay 0,001; LoRA r=16, alpha=16, dropout 0.',
    batch: 'Batch hiệu dụng 8 (4×2); Fable-300: 1 epoch ~113 bước; Fable-1000: 3 epoch 339 bước; tập 900/100.',
    hardware: 'Tesla T4; Fable-1000 khoảng 47 phút, Fable-300 khoảng 15 phút.',
    reg: 'Dropout LoRA 0, weight decay 0,001, gradient checkpointing kiểu Unsloth.',
  },
  flow: 'Học: adapter LoRA cập nhật trên chuỗi chat system/user/assistant; Sinh: sinh tuần tự token truyện tới <|eot_id|>.',
});
eProg('E5', 'Llama 3.2 3B + QLoRA',
  'Hai lần chạy trên T4: Fable-300 (1 epoch) và Fable-1000 (3 epoch); trainer_state ghi cửa sổ loss giảm nhanh rồi phẳng.',
  [
    ['Fable-300', '1 epoch, ~113 bước, ~15 phút; loss cuối ~0,492.'],
    ['Fable-1000', '3 epoch, 339 bước, ~47 phút; loss cuối 0,514.'],
    ['Bước 10', 'Loss 1,578; LR đạt đỉnh ~1,98e-4 sau warmup 5 bước.'],
    ['Bước 110', 'Loss 0,492; validation loss ~0,488 tại hết epoch 1.'],
    ['Bước 225', 'Loss 0,428; LR ~6,89e-5 theo lịch giảm tuyến tính.'],
  ],
  fig('figures/tracks/e5_training_curves.png'),
  'Đường huấn luyện QLoRA: train loss giảm nhanh trong epoch đầu rồi phẳng dần qua ba epoch.');
mFlow('E5 · Llama 3.2 3B + QLoRA — Các bước phương pháp', 'TINYSTORY-VN · E5',
  'Mạch nghiên cứu: nạp nền 3B ở 4-bit, cấu hình QLoRA gọn trên attention và MLP, huấn luyện hai lần Fable-300 và Fable-1000, rồi lượng tử hóa GGUF để phục vụ và đánh giá.',
  [
    { id: 'B1', title: 'Chọn nền 3B ở 4-bit', cap: 'Llama 3.2 3B Instruct nạp 4-bit qua Unsloth, chat template, ngữ cảnh 2.048.', phase: 'nền-tảng' },
    { id: 'B2', title: 'Cấu hình QLoRA rank 16', cap: 'r=16, α=16, dropout 0 trên toàn bộ attention + MLP; 24,31M tham số, tức 0,75%.', phase: 'nền-tảng' },
    { id: 'B3', title: 'Chuẩn bị dữ liệu fable', cap: 'ds-tf1-en-3m, 1.000 mẫu JSONL năm trường, chia 900 train / 100 val.', phase: 'nền-tảng' },
    { id: 'B4', title: 'Chạy Fable-300, 1 epoch', cap: 'Ablation ngắn: khoảng 113 bước, ~15 phút T4, loss cuối ~0,492.', phase: 'can-thiệp' },
    { id: 'B5', title: 'Chạy Fable-1000, 3 epoch', cap: '339 bước, ~47 phút GPU T4, loss cuối 0,514, validation loss ~0,451.', phase: 'can-thiệp' },
    { id: 'B6', title: 'Gộp adapter, lượng tử GGUF', cap: 'Merge LoRA vào nền, convert llama.cpp sang GGUF Q4_K_M, model chỉ ~2,0GB.', phase: 'can-thiệp' },
    { id: 'B7', title: 'Phục vụ Ollama và đánh giá', cap: 'Đăng ký các biến thể vào Compare Mode; Quick Eval với giám khảo Llama 3.2 3B.', phase: 'phát-hiện' },
    { id: 'B8', title: 'Kết quả vòng chung', cap: 'Vòng chung 8,44; counterfactual 10/10 đổi diễn biến đúng hướng, chỉ 0,75% tham số.', phase: 'kết-luận' },
  ]);
eResult('E5', 'Llama 3.2 3B + QLoRA',
  [['8,44', 'Điểm vòng chung — mạnh nhất trong các mô hình sinh trực tiếp một lượt.', C.green],
   ['10 / 10', 'Số cặp counterfactual đổi diễn biến đúng hướng khi thay một điều kiện (9,40/10).', C.blue],
   ['0,75%', 'Tỷ lệ tham số được huấn luyện; 339 bước, 47,3 phút trên một GPU T4.', C.gray]],
  'Với chi phí thích nghi tối thiểu, QLoRA tạo được một artifact sinh trực tiếp dùng điều kiện để thay đổi diễn biến. Đây là bằng chứng mạnh nhất trong nhóm về khả năng kết hợp điều kiện ngay trong một lượt sinh.',
  null, null,
  'Giới hạn: chưa cô lập ảnh hưởng của mô hình nền so với QLoRA, hoặc một so với ba epoch; loss chấm trên toàn chuỗi chat, không riêng token truyện.');

// ============================ 20b. CÔNG NGHỆ GIÁM KHẢO TỔNG ============================
{
  const s = p.addSlide(); chrome(s, 'Công nghệ chấm điểm vòng chung cho năm mô hình', { eyebrow: 'TINYSTORY-VN · ĐÁNH GIÁ CHUNG' });
  s.addText('Đánh giá chung dùng một giám khảo LLM thống nhất, chấm mù trên cùng bộ đề, tách biệt với các giám khảo nội bộ của từng hướng; chỉ điểm vòng chung mới dùng để so sánh năm hệ thống.',
    { x: 0.44, y: 1.68, w: 12.45, h: 0.5, fontFace: F, fontSize: 13, italic: true, color: C.gray, lineSpacingMultiple: 1.0 });
  // pipeline 3 bước
  const pc = [0.44, 4.74, 9.04], PW = 3.85, PH = 1.42, py = 2.28;
  const step = (i, col, num, title, body) => {
    const x = pc[i];
    s.addShape(p.ShapeType.roundRect, { x, y: py, w: PW, h: PH, rectRadius: 0.07, fill: { color: C.white }, line: { color: col, width: 1.5 } });
    s.addShape(p.ShapeType.roundRect, { x, y: py, w: 0.12, h: PH, rectRadius: 0.02, fill: { color: col }, line: { type: 'none' } });
    s.addText([{ text: num + '  ', options: { bold: true, color: col } }, { text: title, options: { bold: true, color: C.ink } }],
      { x: x + 0.24, y: py + 0.1, w: PW - 0.4, h: 0.32, fontFace: F, fontSize: 12 });
    s.addText(body, { x: x + 0.24, y: py + 0.48, w: PW - 0.4, h: 0.86, fontFace: F, fontSize: 9.8, color: C.gray, lineSpacingMultiple: 0.98 });
  };
  step(0, C.teal, '1', 'Sinh đầu ra', 'Năm hệ đại diện chạy cùng 25 đề mới = 125 truyện. seed 5410 + chỉ số đề, temperature 0,7, top-p 0,9, repetition penalty 1,1, tối đa 400 token, 1 mẫu.');
  step(1, C.blue, '2', 'Làm mù (blind)', 'Xáo 125 cặp (seed 20260726) thành B001–B125. Giám khảo chỉ thấy đề và truyện, ẩn mô hình, backend, latency và cờ repair; ánh xạ để riêng.');
  step(2, C.green, '3', 'Chấm bằng Gemma', 'gemma-4-26b-a4b-it qua Google GenAI, temperature 0, seed 20260726, thinking MINIMAL; trả JSON bốn số nguyên 1–10 cho mỗi truyện.');
  const arw = { color: C.mute, width: 2, endArrowType: 'triangle' };
  s.addShape(p.ShapeType.line, { x: pc[0] + PW + 0.03, y: py + PH / 2, w: 0.33, h: 0, line: arw });
  s.addShape(p.ShapeType.line, { x: pc[1] + PW + 0.03, y: py + PH / 2, w: 0.33, h: 0, line: arw });
  // cột trái: rubric 4 trục
  s.addText('Rubric bốn trục (JSON, mỗi trục 1 đến 10)', { x: 0.44, y: 4.02, w: 6.1, h: 0.32, fontFace: F, fontSize: 13, bold: true, color: C.teal });
  const rub = [
    ['grammar', 'Ngữ pháp, mạch lạc, văn phong phù hợp trẻ em'],
    ['creativity', 'Độ mới và sức sống của truyện'],
    ['moral_clarity', 'Bài học rõ và được diễn biến chứng minh'],
    ['prompt_adherence', 'Bám character, setting, challenge, outcome, teaching và dòng Moral'],
  ].map(([k, v]) => [
    { text: k, options: { fontFace: 'Courier New', fontSize: 10, bold: true, color: C.teal, valign: 'middle' } },
    { text: v, options: { fontFace: F, fontSize: 10.5, color: C.ink, valign: 'middle' } },
  ]);
  s.addTable(rub, { x: 0.44, y: 4.42, w: 6.1, colW: [1.85, 4.25], border: { type: 'solid', color: C.line, pt: 0.5 }, rowH: 0.5, valign: 'middle', margin: [3, 5, 3, 5] });
  vline(s, 6.72, 4.02, 2.35);
  // cột phải: tổng hợp + thống kê
  s.addText('Tổng hợp điểm và thống kê', { x: 6.95, y: 4.02, w: 5.95, h: 0.32, fontFace: F, fontSize: 13, bold: true, color: C.teal });
  s.addText(bullets([
    'Điểm mỗi truyện = trung bình bốn trục; điểm mỗi hệ = trung bình 25 truyện (kể cả hai đầu ra gần rỗng của E1, Gemma chấm 1).',
    'Khoảng tin cậy 95% dùng bootstrap 10.000 lần theo 25 đề; so sánh bắt cặp lấy điểm tổng trên cùng đề (E4 − E5 = +0,76 [0,39; 1,17]).',
    'Đây là lượt chấm mới, tách khỏi giám khảo nội bộ (Qwen của E1/E3, Gemma local của E2); điểm nội bộ không nhập vào bảng chung.',
  ]).map(o => ({ ...o, options: { ...o.options, fontSize: 11.5 } })), { x: 6.95, y: 4.42, w: 5.95, h: 2.0 });
  // dải dưới: artifact và môi trường
  s.addShape(p.ShapeType.roundRect, { x: 0.44, y: 6.5, w: 12.0, h: 0.5, rectRadius: 0.05, fill: { color: C.panel }, line: { color: C.line, width: 0.75 } });
  s.addText([{ text: 'Artifact vòng chung:  ', options: { bold: true, color: C.teal } },
    { text: 'E1/E5 GGUF qua llama.cpp · E2 MLX-LM · E3 Transformers/PEFT · E4 llama.cpp kèm validator/repair; chạy trên Apple M4 Pro 24 GiB.', options: { color: C.ink } }],
    { x: 0.62, y: 6.5, w: 11.7, h: 0.5, fontFace: F, fontSize: 10.5, valign: 'middle' });
}

// ============================ 21. BẢNG ĐỊNH LƯỢNG ============================
{
  const s = p.addSlide(); chrome(s, 'Kết quả định lượng trên 25 đề đánh giá thống nhất');
  const data = [{ name: 'Điểm vòng chung', labels: ['E4 · 3B+Repair', 'E5 · 3B QLoRA', 'E1 · 60M', 'E2 · 63M', 'E3 · 135M'], values: [9.2, 8.44, 3.3, 3.18, 2.81] }];
  s.addChart(p.ChartType.bar, data, {
    x: 0.44, y: 1.9, w: 7.6, h: 4.8, barDir: 'col',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelColor: C.ink, dataLabelFontSize: 13, dataLabelFontBold: true,
    chartColors: [C.green, C.green, C.blue, C.blue, C.blue],
    showLegend: false, showTitle: false,
    catAxisLabelColor: C.ink, catAxisLabelFontSize: 12,
    valAxisHidden: true, valGridLine: { style: 'none' }, catGridLine: { style: 'none' },
    valAxisMaxVal: 10, valAxisMinVal: 0,
  });
  stat(s, 8.4, 2.0, '+0,76', 'Chênh lệch E4 − E5 (giám khảo Gemma), khoảng tin cậy [0,39; 1,17].', C.blue, 4.4);
  stat(s, 8.4, 3.35, '25 đề · 125 truyện', 'Sinh lại đầu ra hệ thống đại diện, chấm mù 125 lượt bằng cùng một giám khảo.', C.gray, 4.4);
  s.addText('Trung bình của E4 phản ánh cả mô hình 3B lẫn chi phí hệ thống (kiểm tra, viết lại, chuẩn hóa). Chênh lệch điểm giữa các hướng không thể quy trực tiếp cho số tham số, vì năm hướng còn khác nhau về dữ liệu, giao diện điều kiện và hậu xử lý. Cùng một seed và một giám khảo — chưa phải thang điểm tuyệt đối.',
    { x: 8.4, y: 4.85, w: 4.4, h: 2.0, fontFace: F, fontSize: 12.5, color: C.ink, lineSpacingMultiple: 1.05 });
}

// ============================ 22. BA LỚP KHÔNG THAY THẾ ============================
{
  const s = p.addSlide(); chrome(s, 'Ba lớp bằng chứng không thể thay thế lẫn nhau');
  const lay = (x, tag, sub, body) => {
    s.addShape(p.ShapeType.roundRect, { x, y: 1.9, w: 3.95, h: 3.9, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    s.addText(tag, { x: x + 0.25, y: 2.15, w: 3.5, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.blue });
    s.addText(sub, { x: x + 0.25, y: 2.6, w: 3.5, h: 0.35, fontFace: F, fontSize: 11.5, italic: true, color: C.gray });
    s.addText(body, { x: x + 0.25, y: 3.05, w: 3.5, h: 2.6, fontFace: F, fontSize: 12.5, color: C.ink, lineSpacingMultiple: 1.06 });
  };
  lay(0.44, 'Độ giống dữ liệu', 'Loss · PPL · fluency', 'E1, E2 và E3 đều cải thiện nội bộ nhưng vẫn có thể không bám yêu cầu khi chuyển sang giao diện chung.');
  lay(4.69, 'Mức độ dùng điều kiện', 'Coverage · chuỗi nhân quả · counterfactual', 'E5 đổi diễn biến đúng hướng; E2 nhận biết quan hệ truyện–bài học nhưng khi sinh vẫn chưa tự lập kế hoạch.');
  lay(8.94, 'Độ tin cậy pipeline', 'Validator · postprocess · repair', 'E4 sửa định dạng và lỗi cục bộ; repair không tự tạo ra quan hệ nhân quả trong truyện.');
  s.addText('Điểm overall chỉ có ý nghĩa khi ba lớp được báo cáo và diễn giải riêng; một con số duy nhất che mất ba loại năng lực khác nhau.',
    { x: 0.44, y: 6.05, w: 12.4, h: 0.6, fontFace: F, fontSize: 14, italic: true, color: C.gray, align: 'center' });
}

// ============================ 23. KẾT LUẬN ============================
{
  const s = p.addSlide(); s.background = { color: C.white };
  PAGE += 1;
  s.addText('KẾT LUẬN', { x: 0.7, y: 0.7, w: 6, h: 0.4, fontFace: F, fontSize: 15, color: C.gray, charSpacing: 1 });
  if (HAS_LOGO) s.addImage({ path: LOGO, x: 11.0, y: 0.6, w: 1.8, h: 0.76, sizing: { type: 'contain', w: 1.8, h: 0.76 } });
  s.addText([
    { text: 'Điều kiện phải chi phối diễn biến,', options: { breakLine: true } },
    { text: 'không chỉ xuất hiện trong truyện.', options: {} },
  ], { x: 0.7, y: 1.9, w: 11.8, h: 1.7, fontFace: FL, fontSize: 42, bold: true, color: C.ink, lineSpacingMultiple: 1.02 });
  s.addText(bullets([
    'Đóng góp chính không nằm ở nhận định "mô hình lớn điểm cao hơn", mà ở việc phân biệt ba mức tuân thủ: nhắc lại trường, tổ chức diễn biến theo toàn bộ điều kiện, và thay đổi diễn biến đúng hướng khi can thiệp một điều kiện.',
    'E4 là hệ thống đáng tin cậy nhất khi nội dung gốc đã đúng ngữ nghĩa; E5 là mô hình sinh trực tiếp mạnh nhất về khả năng kết hợp điều kiện trong một lượt.',
    'Hai kết quả dẫn tới hai ưu tiên thiết kế khác nhau: cải thiện mô hình/dữ liệu/mục tiêu điều kiện hóa khi diễn biến chưa bị điều kiện chi phối; dùng lớp repair khi diễn biến đã đúng nhưng định dạng chưa ổn định.',
  ]), { x: 0.7, y: 4.0, w: 11.9, h: 2.3 });
  s.addText('Hạn chế: 25 đề, một seed, một giám khảo. Bước tiếp theo: chấm nhân lực chung, nhiều seed, định dạng prompt thống nhất và thí nghiệm loại trừ base–checkpoint bắt cặp.',
    { x: 0.7, y: 6.5, w: 11.9, h: 0.6, fontFace: F, fontSize: 12.5, italic: true, color: C.gray });
  s.addText(String(PAGE), { x: 12.5, y: 6.95, w: 0.5, h: 0.3, fontFace: F, fontSize: 10.5, color: C.gray, align: 'right' });
}

const OUT = path.join(ROOT, 'output/presentation/tinystory-vn-summary.pptx');
p.writeFile({ fileName: OUT }).then(() => console.log('  ✓ đã ghi', OUT, '| logo:', HAS_LOGO ? 'CÓ' : 'CHƯA (chừa chỗ)', '| tổng slide:', PAGE));

'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';
import SortDropdown, { SortMode } from '@/components/SortDropdown';
import { sortByPHashSimilarity } from '@/utils/phash';
import CaptionServiceStatus from '@/components/CaptionServiceStatus';

interface ImageEntry {
  img_path: string;
  mtime: number;
  phash?: string;
}

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<ImageEntry[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [sortMode, setSortMode] = useState<SortMode>('filename');
  const [hashMap, setHashMap] = useState<Record<string, string>>({});
  const [hashLoading, setHashLoading] = useState(false);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    console.log('Fetching images for dataset:', dbName);
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        console.log('Images:', data.images);
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);

  // Fetch hashes when similarity mode is selected
  useEffect(() => {
    if (sortMode === 'similarity' && Object.keys(hashMap).length === 0) {
      setHashLoading(true);
      apiClient
        .post('/api/datasets/computeHashes', { datasetName })
        .then((res: any) => {
          setHashMap(res.data.hashes || {});
        })
        .catch(error => {
          console.error('Error computing hashes:', error);
        })
        .finally(() => {
          setHashLoading(false);
        });
    }
  }, [sortMode, datasetName]);

  const sortedImgList = useMemo(() => {
    const list: ImageEntry[] = imgList.map(img => ({
      ...img,
      phash: hashMap[img.img_path],
    }));

    switch (sortMode) {
      case 'filename':
        return [...list].sort((a, b) => a.img_path.localeCompare(b.img_path));
      case 'date':
        return [...list].sort((a, b) => b.mtime - a.mtime);
      case 'similarity':
        if (hashLoading || Object.keys(hashMap).length === 0) {
          return [...list].sort((a, b) => a.img_path.localeCompare(b.img_path));
        }
        return sortByPHashSimilarity(list);
      default:
        return list;
    }
  }, [imgList, sortMode, hashMap, hashLoading]);

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, imgList.length]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {datasetName}</h1>
        </div>
        <div className="flex-1"></div>
        <CaptionServiceStatus />
        <div className="w-px h-6 bg-gray-700 mx-2"></div>
        <SortDropdown value={sortMode} onChange={setSortMode} loading={hashLoading} />
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {PageInfoContent}
        {status === 'success' && sortedImgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {sortedImgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                onDelete={() => refreshImageList(datasetName)}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      <FullscreenDropOverlay
        datasetName={datasetName}
        onComplete={() => refreshImageList(datasetName)}
      />
    </>
  );
}
